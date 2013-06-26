from PyQt4.QtCore import SIGNAL
from PyQt4.QtGui import QTextEdit,QPushButton,QGridLayout,QTextCursor,QDialog,QMessageBox

class TextDialog(QDialog):
    """ QDialog object to edit text by using an editor window.
    """
    def __init__(self, parent=None, title="Edit text...", text="", readonly=False, help=None):
        super(TextDialog,self).__init__(parent)
        self.setWindowTitle(title)
        self.resize(600,500)
        self.text=text
        self.help=help
        self.ok = QPushButton('Ok', self)
        self.connect(self.ok, SIGNAL('clicked()'), self.accept)
        if not readonly:
            self.cancel = QPushButton('Cancel', self)
            self.connect(self.cancel, SIGNAL('clicked()'), self.reject)
        if help:
            self.helpButton = QPushButton('Help', self)
            self.connect(self.helpButton, SIGNAL('clicked()'), self.showHelp)
        self.edit=QTextEdit()
        self.edit.setPlainText(self.text)
        layout=QGridLayout()
        layout.addWidget(self.edit,0,0,1,4)
        layout.addWidget(self.ok,1,3)
        if not readonly:
            layout.addWidget(self.cancel,1,0)
        if help:
            layout.addWidget(self.helpButton,1,1)
        self.setLayout(layout)
        self.edit.setReadOnly(readonly)
        self.edit.setFocus()
        self.edit.moveCursor(QTextCursor.End)

    def getText(self):
        return self.edit.toPlainText().toAscii()

    def showHelp(self):
        QMessageBox.about(self, 'Info', self.help)
