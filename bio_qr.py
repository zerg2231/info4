import cv2
import numpy as np
import qrcode as qrc
import sys

from matplotlib import pyplot as plt
from PIL import Image as img
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(390, 160)
        MainWindow.setMinimumSize(QtCore.QSize(390, 160))
        MainWindow.setMaximumSize(QtCore.QSize(390, 160))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 321, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 40, 311, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 60, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(170, 60, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 110, 221, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(240, 110, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.button_click)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 130, 221, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Программа"))
        self.label.setText(_translate("MainWindow", "Запись QR-кода в изображение лица"))
        self.label_2.setText(_translate("MainWindow", "Введите индекс человека из базы данных"))
        self.label_3.setText(_translate("MainWindow", "CUHK (188 классов):"))
        self.label_4.setText(_translate("MainWindow", "Формирование BIO QR-кодов"))
        self.pushButton.setText(_translate("MainWindow", "Сформировать"))
        self.label_5.setText(_translate("MainWindow", "(в нескольких вариантах):"))
    
    def button_click(self):
        QR(self.lineEdit.text()).show_results()

class Image():
    def __init__(self, i, tp):
        self.tp = tp
        self.image = cv2.imread(f'cuhk\photos\{i}.jpg')
        self.image_gray = cv2.imread(f'cuhk\photos\{i}.jpg', 0)
        self.sketch = cv2.imread(f'cuhk\sketches\{i}.jpg')
        self.sketch_gray = cv2.imread(f'cuhk\sketches\{i}.jpg', 0)
        
    def get_image(self):
        if self.tp == 'i_normal':
            return self.image
        
        elif self.tp == 'i_normal_r':
            return self.image[25:225, 0:200]
        
        elif self.tp == 'i_gray_r':
            return self.image_gray[25:225, 0:200]
        
        elif self.tp == 's_normal':
            return self.sketch
        
        elif self.tp == 's_normal_r':
            return self.sketch[25:225, 0:200]
        
        elif self.tp == 's_gray_r':
            return self.sketch_gray[25:225, 0:200]
        
        elif self.tp == 'index':
            return self.i
    
class QR():
    def __init__(self, i):
        self.i = i
        self.image_normal = Image(self.i, 'i_normal').get_image()
        self.sketch_normal = Image(self.i, 's_normal').get_image()
        self.image = Image(self.i, 'i_normal_r').get_image()
        self.image_gray = Image(self.i, 'i_gray_r').get_image()
        self.sketch = Image(self.i, 's_normal_r').get_image()
        self.sketch_gray = Image(self.i, 's_gray_r').get_image()
        self.i_r, self.i_g, self.i_b = img.fromarray(self.image).split()
        self.s_r, self.s_g, self.s_b = img.fromarray(self.sketch).split()
    
    def generate(self, data):
        qr = qrc.QRCode(version=1,
                           error_correction=qrc.constants.ERROR_CORRECT_L,
                           box_size=5, border=5)
        qr.add_data(data)
        qr.make(fit=True)
        
        image = qr.make_image(fill_color="black", back_color="white")
        
        return image.resize((200, 200))
    
    def antro(self, tp):
        haarcascade = 'haarcascades/haarcascade_frontalface_alt2.xml'
        detector = cv2.CascadeClassifier(haarcascade)
        face = detector.detectMultiScale(self.image_gray, 1.3, 5)
        
        LBFmodel = 'facemark_api/lbfmodel.yaml'
        landmark_detector  = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(LBFmodel)
        _, landmarks = landmark_detector.fit(self.image_gray, face)
        
        if tp == 'data':
            return landmarks
        
        elif tp == 'image':
            image = np.zeros((200, 200), np.uint8)
            for landmark in landmarks:
                for x, y in landmark[0]:
                    cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 1)
            
            return image[50:200, 25:175]
                    
    
    def pheno(self):
        hist = []
        hist.append(cv2.calcHist([self.image], [0], None, [16], [0, 256]))
        hist.append(cv2.calcHist([self.image], [1], None, [16], [0, 256]))
        hist.append(cv2.calcHist([self.image], [2], None, [16], [0, 256]))
        
        return hist
    
    def info(self):
        text = f'CUHK Face Sketch Database (CUFS):\nclass index = {self.i}.'
        
        return text
    
    def bio(self, tp):
        if tp == 'pip':
            self.i_g.paste(self.generate(self.info()))
        
        elif tp == 'pia':
            self.i_g.paste(self.generate(self.info()))
            self.i_b.paste(self.generate(self.antro('data')))

        elif tp == 'ppi':
            self.i_g.paste(self.generate(self.pheno()))
            self.i_b.paste(self.generate(self.info()))
        
        elif tp == 'ppa':
            self.i_g.paste(self.generate(self.pheno()))
            self.i_b.paste(self.generate(self.antro('data')))
            
        elif tp == 'pis':
            self.i_g.paste(self.generate(self.info()))
            self.i_b = self.s_b
        
        qr = img.merge('RGB', (self.i_b, self.i_g, self.i_r))
        
        return qr
    
    def show_results(self):
        fig = plt.figure('Результат ', figsize=(16, 8))
        
        ax1 = fig.add_subplot(2, 4, 1)
        ax2 = fig.add_subplot(2, 4, 5)
        ax3 = fig.add_subplot(2, 4, 2)
        ax4 = fig.add_subplot(2, 4, 3)
        ax5 = fig.add_subplot(2, 4, 4)
        ax6 = fig.add_subplot(2, 4, 6)
        ax7 = fig.add_subplot(2, 4, 7)
        
        ax1.imshow(cv2.cvtColor(self.image_normal, cv2.COLOR_BGR2RGB))
        ax1.axis("off")
        ax1.set_title('Photo')
        
        ax2.imshow(cv2.cvtColor(self.sketch_normal, cv2.COLOR_BGR2RGB))
        ax2.axis("off")
        ax2.set_title('Sketch')
        
        ax3.imshow(self.bio('pip'))
        ax3.axis("off")
        ax3.set_title('BIO QR: PIP\n(Photo/INFO/Photo)')
        
        ax4.imshow(self.bio('pia'))
        ax4.axis("off")
        ax4.set_title('BIO QR: PIA\n(Photo/INFO/ANTRO)')
        
        ax5.imshow(self.bio('ppi'))
        ax5.axis("off")
        ax5.set_title('BIO QR: PPI\n(Photo/PHENO/INFO)')
        
        ax6.imshow(self.bio('ppa'))
        ax6.axis("off")
        ax6.set_title('BIO QR: PPA\n(Photo/PHENO/ANTRO)')
        
        ax7.imshow(self.bio('pis'))
        ax7.axis("off")
        ax7.set_title('BIO QR: PIS\n(Photo/INFO/Sketch)')
        
        plt.subplots_adjust(wspace=0.5, hspace=0.3,
                            top=0.9, bottom=0.05,
                            left=0.05, right=0.95)
        plt.show()

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()
sys.exit(app.exec_())
