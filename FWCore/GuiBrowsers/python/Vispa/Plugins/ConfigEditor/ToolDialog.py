import logging
import copy

from PyQt4.QtCore import QCoreApplication,Qt,SIGNAL
from PyQt4.QtGui import QDialog,QListWidget,QVBoxLayout,QHBoxLayout,QToolButton,QPushButton,QSplitter

from Vispa.Main.Application import Application
from Vispa.Views.PropertyView import PropertyView
from ToolDataAccessor import toolsDict
from Vispa.Main.Exceptions import exception_traceback

class ToolDialog(QDialog):
    def __init__(self,parent=None):
        logging.debug(__name__ +': __init__')
        QDialog.__init__(self,parent)
        self.resize(600,500)
        self._selectedTool=None
        self._processCopy=None
        self._configDataAccessor=None
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
        self.connect(ok, SIGNAL('clicked()'), self.apply)
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
        self._toolDataAccessor=accessor
        # save process copy to undo changes during the tool dialog
        self._processCopy=copy.deepcopy(self._toolDataAccessor.configDataAccessor().process())

    def apply(self):
        parameterErrors=self._toolDataAccessor.parameterErrors(self._selectedTool)
        if len(parameterErrors)>0:
            ok=False
            message="\n".join([error for error in parameterErrors])
            QCoreApplication.instance().errorMessage(message)
            return
        ok=True
        try:
            self._selectedTool.apply(self._toolDataAccessor.configDataAccessor().process())
        except Exception,e:
            ok=False
            logging.error(__name__ + ": Cannot apply tool: "+exception_traceback())
            QCoreApplication.instance().errorMessage("Cannot apply tool (see log file for details):\n"+str(e))
        # recover process copy to undo changes during the tool dialog
        self._toolDataAccessor.configDataAccessor().setProcess(self._processCopy)
        if ok:
            self.accept()
