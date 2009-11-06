import logging

from PyQt4.QtCore import QCoreApplication,Qt,SIGNAL
from PyQt4.QtGui import QDialog,QListWidget,QVBoxLayout,QHBoxLayout,QToolButton,QPushButton,QSplitter

from Vispa.Main.Application import Application
from Vispa.Views.PropertyView import PropertyView
from ToolDataAccessor import toolsDict

class ToolDialog(QDialog):
    def __init__(self,parent=None):
        logging.debug(__name__ +': __init__')
        QDialog.__init__(self,parent)
        self.resize(600,500)
        self._selectedTool=None
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Apply PAT tool...")
        self.fill()
        
    def fill(self):
        logging.debug(__name__ +': fill')
        self.setLayout(QVBoxLayout())

        self._splitter=QSplitter()
        self.layout().addWidget(self._splitter)

        self._toolList = QListWidget(self._splitter)
        for item in toolsDict.keys():
            self._toolList.addItem(item)
        self._toolList.sortItems()
        self.connect(self._toolList, SIGNAL("itemSelectionChanged()"), self.toolSelected)
        self._properties=PropertyView(self._splitter)

        bottom=QHBoxLayout()
        self.layout().addLayout(bottom)
        bottom.addStretch()
        ok=QPushButton("&Apply")
        bottom.addWidget(ok)
        self.connect(ok, SIGNAL('clicked()'), self.accept)
        cancel = QPushButton('&Cancel')
        bottom.addWidget(cancel)
        self.connect(cancel, SIGNAL('clicked()'), self.reject)

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_W:
            self.reject()
        QDialog.keyPressEvent(self, event)

    def tool(self):
        return self._selectedTool

    def toolSelected(self):
        self._selectedTool=toolsDict[str(self._toolList.currentItem().text())]()
        self._properties.setDataObject(self._selectedTool)
        self._properties.updateContent()

    def setDataAccessor(self,accessor):
        self._properties.setDataAccessor(accessor)
