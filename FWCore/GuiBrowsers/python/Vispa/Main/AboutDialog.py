from PyQt4.QtCore import Qt, SIGNAL, SLOT
from PyQt4.QtGui import QDialog, QPalette, QVBoxLayout, QLabel, QPushButton, QDialogButtonBox, QSizePolicy
from PyQt4.QtSvg import QSvgWidget 

from Vispa.Main.Directories import websiteUrl
import logging

class AboutDialog(QDialog):
    def __init__(self, application):
        self._application = application
        QDialog.__init__(self, self._application.mainWindow())
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setWindowFlags(Qt.Window)
        self.setAutoFillBackground(True)
        #self.setPalette(QPalette(Qt.white))
        self.fill()
        self.setWindowTitle("About "+self._application.windowTitle())
        self.resize(220, 200)
        
    def setApplication(self, app):
        self._application = app
        
    def fill(self):
        # Vispa 
        # icon
        # Version
        # Release date os.path.getmtime(filename)
        # website url
        # license
        self.setLayout(QVBoxLayout())
        if "vispa" in self._application.windowTitle().lower():
            self._logo = QSvgWidget(":/resources/vispa_logo.svg")
            sizeHint = self._logo.sizeHint()
            logo_width_height_ratio =  1.0 * sizeHint.width() / sizeHint.height()
            logo_width = 200
            self._logo.setFixedSize(logo_width, logo_width/logo_width_height_ratio)
            self.layout().addWidget(self._logo)
        else:
            label=QLabel(self._application.windowTitle())
            self.layout().addWidget(label)
        self.layout().addWidget(QLabel("Version "+ self._application.version()))
        self.layout().addWidget(QLabel("More information can be found on:"))
        websiteLink = QLabel("<a href='"+ websiteUrl +"'>"+ websiteUrl +"</a>")
        websiteLink.setTextInteractionFlags(Qt.LinksAccessibleByMouse | Qt.TextSelectableByMouse)
        websiteLink.setOpenExternalLinks(True)
        self.layout().addWidget(websiteLink)
        buttonBox=QDialogButtonBox()
        buttonBox.addButton(QDialogButtonBox.Close)
        self.connect(buttonBox,SIGNAL("rejected()"),self,SLOT("reject()"))
        self.layout().addWidget(buttonBox)

    def onScreen(self):
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()
