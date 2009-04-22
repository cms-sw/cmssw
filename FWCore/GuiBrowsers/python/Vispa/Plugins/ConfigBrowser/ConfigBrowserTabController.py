import logging

from Vispa.Main.Exceptions import *
from Vispa.Main.Thread import *
from Vispa.Main.TripleTabController import *
from DOTExport import *

class ConfigBrowserTabController(TripleTabController):
    """
    """
    def __init__(self, plugin):
        logging.debug(__name__ + ": __init__")
        TripleTabController.__init__(self, plugin)

        self._editorName = ""
        self._thread = None
        
        self._viewMenu.addSeparator()
        self._sequenceStructureAction = self.plugin().application().createAction('&Sequence structure', self.sequenceStructure, "Ctrl+U")
        self._sequenceStructureAction.setCheckable(True)
        self.viewMenu().addAction(self._sequenceStructureAction)
        self._connectionStructureAction = self.plugin().application().createAction('&Connection structure', self.connectionStructure, "Ctrl+N")
        self._connectionStructureAction.setCheckable(True)
        self._connectionStructureAction.setChecked(True)
        self.viewMenu().addAction(self._connectionStructureAction)
        self._configMenu = self.plugin().application().createPluginMenu('&Config')
        refreshAction = self.plugin().application().createAction('&Refresh', self.refresh, 'F5')
        self._configMenu.addAction(refreshAction)
        openEditorAction = self.plugin().application().createAction('&Open editor...', self.openEditor, "F6")
        self._configMenu.addAction(openEditorAction)
        chooseEditorAction = self.plugin().application().createAction('&Choose editor...', self.chooseEditor, "Ctrl+T")
        self._configMenu.addAction(chooseEditorAction)
        self._configMenu.addSeparator()
        self._readOnlyAction = self.plugin().application().createAction('&Read only mode', self.readOnlyMode, "Ctrl+R")
        self._readOnlyAction.setCheckable(True)
        self._configMenu.addAction(self._readOnlyAction)
        self._dumpAction = self.plugin().application().createAction('&Dump python config to single file...', self.dumpPython, "Ctrl+D")
        self._configMenu.addAction(self._dumpAction)
        self._configMenu.addSeparator()
        self._dotExportAction = self.plugin().application().createAction('&Export dot graphic...', self.exportDot, "Ctrl+G")
        self._configMenu.addAction(self._dotExportAction)
        
    #@staticmethod
    def staticSupportedFileTypes():
        """ Returns supported file type: py.
        """
        return [('py', 'Config file')]
    staticSupportedFileTypes = staticmethod(staticSupportedFileTypes)
    
    def readOnlyMode(self):
        self.tab().propertyView().setReadOnly(self._readOnlyAction.isChecked())
        self.setEditable(not self._readOnlyAction.isChecked())
        self.tab().propertyView().updateContent()
        
    def readOnlyAction(self):
        return self._readOnlyAction
    
    def dotExportAction(self):
        return self._dotExportAction
    
    def dumpAction(self):
        return self._dumpAction
    
    def refresh(self):
        if self.allowClose():
            self.plugin().application().mainWindow().statusBar().showMessage("Reopening config...")
            thread = RunThread(self.dataAccessor().open, self._filename)
            while thread.isRunning():
                qApp.processEvents()
            self.updateAndRestoreSelection()
            self.plugin().application().mainWindow().statusBar().showMessage("Reopening config...done")
    
    def sequenceStructure(self):
        """ Show the sequence structure in center view.
        """
        self._connectionStructureAction.setChecked(not self._sequenceStructureAction.isChecked())
        self.updateAndRestoreSelection()
        self._saveIni()
    
    def connectionStructure(self):
        """ Show the connection structure in center view.
        """
        self._sequenceStructureAction.setChecked(not self._connectionStructureAction.isChecked())
        self.updateAndRestoreSelection()
        self._saveIni()

    def updateCenterView(self, item):
        """ Fill the center view from an item in the TreeView and update it """
        self.plugin().application().mainWindow().statusBar().showMessage("Updating center view...")
        if self._thread != None and self._thread.isRunning():
            self.dataAccessor().cancelOperations()
            while self._thread.isRunning():
                qApp.processEvents()
        objects = []
        if item != None:
            if self._sequenceStructureAction.isChecked():
                objects = [item.object]
            else:
                objects = self.dataAccessor().nonSequenceChildren(item.object)
        self.tab().centerView().setDataObjects(objects)
        if self._connectionStructureAction.isChecked():
            thread = RunThread(self.dataAccessor().readConnections, objects)
            self._thread = thread
            while thread.isRunning():
                qApp.processEvents()
            if thread.returnValue:
                self.tab().centerView().setConnections(self.dataAccessor().connections())
        else:
            self.tab().centerView().setConnections([])
        self.tab().centerView().updateContent()
        self.plugin().application().mainWindow().statusBar().showMessage("Updating center view...done")

    def selected(self):
        """ Shows plugin menus when user selects tab.
        """
        logging.debug(__name__ + ": selected()")
        TripleTabController.selected(self)
        self.plugin().application().showPluginMenu(self._configMenu)

    def openEditor(self):
        """ Call editor """
        logging.debug(__name__ + ": openEditor")
        selected_object = self.tab().propertyView().dataObject()
        filename = self.dataAccessor().fullFilename(selected_object)
        if self._editorName != "" and selected_object != None and os.path.exists(filename):
            if os.path.expandvars("$CMSSW_RELEASE_BASE") in filename:
                QMessageBox.information(self.tab(), "Opening readonly file...", "This file is from $CMSSW_RELEASE_BASE and readonly") 
            command = self._editorName
            command += " " + filename
            command += " &"
            os.system(command)

    def chooseEditor(self, _editorName=""):
        """ Choose editor using FileDialog """
        logging.debug(__name__ + ": chooseEditor")
        if _editorName == "":
            _editorName = str(QFileDialog.getSaveFileName(self.tab(), "Choose editor", self._editorName, "Editor (*)",None , QFileDialog.DontConfirmOverwrite or QFileDialog.DontResolveSymlinks))
            if not os.path.exists(_editorName):
                _editorName = os.path.basename(_editorName)
        if _editorName != None and _editorName != "":
            self._editorName = _editorName
        self._saveIni()

    def save(self, filename=''):
        (config_path, fileName) = os.path.split(str(filename))
        configName = os.path.splitext(fileName)[0]
        if configName==self.dataAccessor().configName():
            self.plugin().application().errorMessage("Cannot overwrite original configuration file.")
        elif self.dataAccessor().isReplaceConfig():
            return TabController.save(self, filename)
        elif filename != "":
            if TabController.save(self, filename):
                self.dataAccessor().setIsReplaceConfig()
                return True
            return False
        return self.tab().mainWindow().application().saveFileAsDialog()

    def writeFile(self, filename):
        """ Write replace config file.
        """
        logging.debug(__name__ + ': writeFile')
        dump = self.dataAccessor().dumpReplaceConfig()
        text_file = open(filename, "w")
        text_file.write(dump)
        text_file.close()
        return True
    
    def dumpPython(self, fileName=None):
        """ Dump python configuration to file """
        logging.debug(__name__ + ": dumpPython")
        dump = self.dataAccessor().dumpPython()
        if dump == "":
            self.plugin().application().errorMessage("Cannot dump this config because it does not 'process'.\nNote that only 'cfg' files contain a 'process'.)")
            return None
        filter = QString("")
        if not fileName:
            defaultname = os.path.splitext(self._filename)[0] + "_dump" + os.path.splitext(self._filename)[1]
            fileName = str(QFileDialog.getSaveFileName(self.tab(), "Save python config...", defaultname, "Python config (*.py)", filter))
        if fileName != "":
            name = fileName
            ext = "PY"
            if os.path.splitext(fileName)[1].upper().strip(".") == ext:
                name = os.path.splitext(fileName)[0]
                ext = os.path.splitext(fileName)[1].upper().strip(".")
            text_file = open(name + "." + ext.lower(), "w")
            text_file.write(dump)
            text_file.close()

    def _loadIni(self):
        """ read options from ini """
        ini = self.plugin().application().ini()
        if ini.has_option("config", "editor"):
            self._editorName = str(ini.get("editor", "filename"))
        else:
            self._editorName = "emacs"
        if ini.has_option("config", "connection structure"):
            self._connectionStructureAction.setChecked(ini.getboolean("config", "connection structure"))
            self._sequenceStructureAction.setChecked(not ini.getboolean("config", "connection structure"))
        if ini.has_option("config", "box content script"):
            self.tab().centerView().setBoxContentScript(str(ini.get("config", "box content script")))
            self._boxContentDialog.setScript(str(ini.get("config", "box content script")))

    def scriptChanged(self, script):
        TripleTabController.scriptChanged(self, script)
        self._saveIni()

    def _saveIni(self):
        """ write options to ini """
        ini = self.plugin().application().ini()
        if not ini.has_section("config"):
            ini.add_section("config")
        ini.set("config", "editor", self._editorName)
        ini.set("config", "connection structure", self._connectionStructureAction.isChecked())
        ini.set("config", "box content script", self.tab().centerView().boxContentScript())
        self.plugin().application().writeIni()

    def exportDot(self, fileName=None):
        dot = DotExport()
        if self._connectionStructureAction.isChecked():
            presets = {'endpath':False,'source':False,'legend':False}
        else:
            presets = {'seq':True, 'seqconnect':True,'tagconnect':False,'legend':False}
        for opt,val in presets.items():
            dot.setOption(opt,val)
        types=""
        for ft in dot.file_types:
            if types!="":
                types+=";;"
            types+=ft.upper()+" File (*."+ft.lower()+")"
        filter=QString("PDF File (*.pdf)")
        if not fileName:
            defaultname = os.path.splitext(self._filename)[0] + "_export"
            fileName = str(QFileDialog.getSaveFileName(self.tab(),"Export dot graphic...",defaultname,types,filter))
        if fileName!="":
            name=fileName
            ext=str(filter).split(" ")[0].lower()
            if os.path.splitext(fileName)[1].lower().strip(".") in dot.file_types:
                name=os.path.splitext(fileName)[0]
                ext=os.path.splitext(fileName)[1].lower().strip(".")
            try:
                dot.export(self.dataAccessor(),name+"."+ext,ext)
            except Exception:
                self.plugin().application().errorMessage("Could not export dot graphic: " + exception_traceback())

    def setTab(self, tab):
        TripleTabController.setTab(self, tab)
        self._loadIni()
