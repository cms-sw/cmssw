import logging
import copy
import os.path
import imp
import inspect

from PyQt4.QtCore import QCoreApplication,Qt,SIGNAL
from PyQt4.QtGui import QDialog,QListWidget,QVBoxLayout,QHBoxLayout,QToolButton,QPushButton,QSplitter,QFileDialog,QMessageBox

from Vispa.Main.Application import Application
from Vispa.Views.PropertyView import PropertyView
from Vispa.Main.Exceptions import exception_traceback
from ToolDataAccessor import standardToolsDir,ConfigToolBase

class ToolDialog(QDialog):
    def __init__(self,parent=None):
        logging.debug(__name__ +': __init__')
        QDialog.__init__(self,parent)
        self.resize(600,500)
        self._selectedTool=None
        self._processCopy=None
        self._configDataAccessor=None
        self._toolsDir=standardToolsDir
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Apply tool...")
        self.fill()
    
    def fill(self):
        logging.debug(__name__ +': fill')
        self.setLayout(QVBoxLayout())

        self._splitter=QSplitter()
        self.layout().addWidget(self._splitter)

        self._toolList = QListWidget(self._splitter)
        self.connect(self._toolList, SIGNAL("itemSelectionChanged()"), self.toolSelected)
        self._properties=PropertyView(self._splitter)

        bottom=QHBoxLayout()
        self.layout().addLayout(bottom)
        changedir=QPushButton("&Change tools directory...")
        bottom.addWidget(changedir)
        self.connect(changedir, SIGNAL('clicked()'), self.changedir)
        help=QPushButton("&Help")
        bottom.addWidget(help)
        self.connect(help, SIGNAL('clicked()'), self.help)
        bottom.addStretch()
        cancel = QPushButton('&Cancel')
        bottom.addWidget(cancel)
        self.connect(cancel, SIGNAL('clicked()'), self.reject)
        self.ok=QPushButton("&Apply")
        bottom.addWidget(self.ok)
        self.ok.setDefault(True)
        self.connect(self.ok, SIGNAL('clicked()'), self.apply)

    def updateToolList(self):
        self._toolList.clear()
        # import all tools and register them in toolsDict
        toolsFiles = [os.path.join(self._toolsDir,f) for f in os.listdir(self._toolsDir) if f.endswith(".py") and not f.startswith("_")]
        self._toolsDict={}
        for toolsFile in toolsFiles:
            pythonModule = os.path.splitext(os.path.basename(toolsFile))[0]
            module=imp.load_source(pythonModule, toolsFile)
            for name in dir(module):
                tool=getattr(module,name)
                if inspect.isclass(tool) and issubclass(tool,ConfigToolBase) and not self._toolDataAccessor.label(tool) in self._toolsDict.keys() and not tool==ConfigToolBase:
                    self._toolsDict[self._toolDataAccessor.label(tool)]=tool
        # Show test tool
        #from FWCore.GuiBrowsers.editorTools import ChangeSource
        #self._toolsDict["ChangeSource"]=ChangeSource
        if len(self._toolsDict.keys())==0 and self._toolsDir==standardToolsDir:
            logging.error(__name__ + ": Could not find any PAT tools. These will be available for the ConfigEditor in a future release.")
            QCoreApplication.instance().errorMessage("Could not find any PAT tools. These will be available for the ConfigEditor in a future release.")
            return
        for item in self._toolsDict.keys():
            self._toolList.addItem(item)
        self._toolList.sortItems()

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_W:
            self.reject()
        QDialog.keyPressEvent(self, event)

    def tool(self):
        return self._selectedTool

    def toolSelected(self):
        self._selectedTool=self._toolsDict[str(self._toolList.currentItem().text())]()
        self._properties.setDataObject(self._selectedTool)
        self._properties.updateContent()

    def setDataAccessor(self,accessor):
        self._properties.setDataAccessor(accessor)
        self._toolDataAccessor=accessor
        # save process copy to undo changes during the tool dialog
        self._processCopy=copy.deepcopy(self._toolDataAccessor.configDataAccessor().process())
        self.updateToolList()

    def apply(self):
        # allow property view to process parameter changes
        if not self.ok.hasFocus():
            self._properties.clearFocus()
            self.ok.setFocus()
            return
        parameterErrors=self._toolDataAccessor.parameterErrors(self._selectedTool)
        if len(parameterErrors)>0:
            message="\n".join([error for error in parameterErrors])
            QCoreApplication.instance().errorMessage(message)
            return
        ok=True
        if self._selectedTool:
            try:
                self._selectedTool.apply(self._toolDataAccessor.configDataAccessor().process())
                if not self._toolDataAccessor.configDataAccessor().process().checkRecording():
                    ok=False
                    logging.error(__name__ + ": Could not apply tool "+self._toolDataAccessor.label(self._selectedTool)+" (problem with enable recording flag) Please restart the ConfigEditor.")
                    QCoreApplication.instance().errorMessage("Could not apply tool "+self._toolDataAccessor.label(self._selectedTool)+" (problem with enable recording flag) Please restart the ConfigEditor.")
            except Exception,e:
                ok=False
                logging.error(__name__ + ": Could not apply tool "+self._toolDataAccessor.label(self._selectedTool)+": "+exception_traceback())
                QCoreApplication.instance().errorMessage("Cannot apply tool "+self._toolDataAccessor.label(self._selectedTool)+" (see log file for details):\n"+str(e))
        else:
            ok=False
            logging.error(__name__ + ": Could not apply tool: No tool selected.")
            QCoreApplication.instance().errorMessage("Cannot apply tool: No tool selected.")
        # recover process copy to undo changes during the tool dialog
        self._toolDataAccessor.configDataAccessor().setProcess(self._processCopy)
        if ok:
            self.accept()

    def changedir(self):
        filename = QFileDialog.getExistingDirectory(
            self,'Select a directory',self._toolsDir,QFileDialog.ShowDirsOnly)
        if not filename.isEmpty():
            self._toolsDir=str(filename)
            self.updateToolList()

    def help(self):
        QMessageBox.about(self, 'Info', "This dialog let's you choose and configure a tool.\n 1. Select a tool on the left\n 2. Set the parameters on the right. If you hold the mouse over a parameter name a tooltip with a description of the parameter will show up.\n 3. Apply the tool. In case the tool has wrong parameters set an error message will be displayed.")
