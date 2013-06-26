import logging

from PyQt4.QtCore import *
from PyQt4.QtGui import *

class BoxContentDialog(QDialog):
    def __init__(self, parent=None):
        logging.debug(__name__ + ': __init__')
        QDialog.__init__(self, parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Set box content...")

        self._buttons = []
        
        self.fill()
        self.hide()
        
    def fill(self):
        logging.debug(__name__ + ': fill')
        self._scriptLabel = QLabel("Script: ")
        self._scriptTextEdit = QTextEdit()

        self._applyButton = QPushButton("&Apply")
        self._applyButton.setDefault(True)
        self._helpButton = QPushButton("&Help")
        self._cancelButton = QPushButton("&Cancel")

        self.setLayout(QVBoxLayout())
        self._layout3 = QHBoxLayout()
        self._layout4 = QHBoxLayout()
    
        self.layout().addWidget(self._scriptLabel)
        self.layout().addWidget(self._scriptTextEdit)
        self.layout().addStretch()
        self.layout().addLayout(self._layout3)
        self.layout().addLayout(self._layout4)
        
        self._layout4.addWidget(self._cancelButton)
        self._layout4.addWidget(self._helpButton)
        self._layout4.addStretch()
        self._layout4.addWidget(self._applyButton)

        self.connect(self._applyButton, SIGNAL('clicked(bool)'), self.apply)
        self.connect(self._helpButton, SIGNAL('clicked(bool)'), self.help)
        self.connect(self._cancelButton, SIGNAL('clicked(bool)'), self.reject)
        
        self._addLabelLabel = QLabel("Add: ")
        self._layout3.addWidget(self._addLabelLabel)
        self.addButton("&Space", "' '")
        self.addButton("&New line", "'\n'")
        self._layout3.addStretch()

    def addButton(self, name, script):
        button = QPushButton(name)
        button.script = script
        self._buttons += [button]
        self._layout3.addWidget(button)
        self.connect(button, SIGNAL('pressed()'), self.buttonPressed)

    def buttonPressed(self):
        for button in self._buttons:
            if button.isDown():
                if self.script() != "":
                    self._scriptTextEdit.textCursor().insertText("+")
                self._scriptTextEdit.textCursor().insertText(button.script)
        
    def onScreen(self):
        logging.debug(__name__ + ': onScreen')
        self.show()
        self.raise_()
        self.activateWindow()
        self._scriptTextEdit.setFocus()
        
    def keyPressEvent(self, event):
        """ 
        """
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_W:
            self.close()
        QDialog.keyPressEvent(self, event)

    def script(self):
        return str(self._scriptTextEdit.toPlainText().toAscii()).replace("\n", "\\n")
    
    def setScript(self, script):
        self._scriptTextEdit.insertPlainText(script)

    def apply(self):
        self.emit(SIGNAL("scriptChanged"), self.script())
        self.accept()

    def help(self):
        QMessageBox.about(self, 'Info', "This dialog allows you to specify what text shall be displayed inside the boxes of the center view. You can specify any valid Python string or use the buttons to fill the string.")
    