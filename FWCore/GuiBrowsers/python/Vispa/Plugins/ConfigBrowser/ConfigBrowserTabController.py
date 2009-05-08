import sys
import logging
import os.path

from PyQt4.QtCore import SIGNAL
from PyQt4.QtCore import QString
from PyQt4.QtCore import QCoreApplication
from PyQt4.QtGui import QMessageBox
from PyQt4.QtGui import QFileDialog

from Vispa.Main.Exceptions import exception_traceback
from Vispa.Main.Exceptions import PluginIgnoredException
from Vispa.Main.Thread import RunThread
from Vispa.Main.TripleTabController import TripleTabController
from Vispa.Main.Workspace import Workspace
from ConfigBrowserBoxView import ConfigBrowserBoxView

try:
    from DOTExport import DotExport
except Exception:
    raise PluginIgnoredException("cannot import DOTExport: " + exception_traceback())
    pass

class ConfigBrowserTabController(TripleTabController):
    """
    """
    def __init__(self, plugin):
        logging.debug(__name__ + ": __init__")
        TripleTabController.__init__(self, plugin)

        self._editorName = ""
        self._thread = None
        
        self._viewMenu.addSeparator()
        self.addCenterView("&Connection structure", self.connectionStructure, True, "Ctrl+1")
        self.addCenterView("&Sequence structure", self.sequenceStructure, False, "Ctrl+2")
        self._configBrowserBoxView = ConfigBrowserBoxView()
        self.fillCenterViewSelectMenu()

        self._configMenu = self.plugin().application().createPluginMenu('&Config')
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
    
    def sequenceStructure(self):
        """ Show the sequence structure in center view.
        """
        logging.debug(__name__ + ": sequenceStructure")
        self.tab().setCenterView(self._configBrowserBoxView)
        self._saveIni()
    
    def connectionStructure(self):
        """ Show the connection structure in center view.
        """
        logging.debug(__name__ + ": connectionStructure")
        self.tab().setCenterView(self._configBrowserBoxView)
        self._saveIni()

    def updateCenterView(self):
        """ Fill the center view from an item in the TreeView and update it """
        statusMessage = self.plugin().application().startWorking("Updating center view")
        if self._thread != None and self._thread.isRunning():
            self.dataAccessor().cancelOperations()
            while self._thread.isRunning():
                QCoreApplication.instance().processEvents()
        objects = []
        select=self.tab().treeView().selection()
        if select != None:
            if self.currentCenterView() == "&Connection structure":
                objects = self.dataAccessor().nonSequenceChildren(select)
            else:
                objects = [select]
        self.tab().centerView().setDataObjects(objects)
        if self.currentCenterView() == "&Connection structure":
            thread = RunThread(self.dataAccessor().readConnections, objects)
            self._thread = thread
            while thread.isRunning():
                QCoreApplication.instance().processEvents()
            if thread.returnValue:
                if isinstance(self.tab().centerView(), ConfigBrowserBoxView):
                    self.tab().centerView().setConnections(self.dataAccessor().connections())
        else:
            if isinstance(self.tab().centerView(), ConfigBrowserBoxView):
                self.tab().centerView().setConnections([])
        result = self.tab().centerView().updateContent()
        if result:
            self.tab().centerView().restoreSelection()
            select = self.tab().centerView().selection()
            if select != None:
                if self.tab().propertyView().dataObject() != select:
                    self.tab().propertyView().setDataObject(select)
                    self.tab().propertyView().updateContent()
        self.plugin().application().stopWorking(statusMessage)
        return result
        
    def selected(self):
        """ Shows plugin menus when user selects tab.
        """
        logging.debug(__name__ + ": selected()")
        TripleTabController.selected(self)
        self.plugin().application().showPluginMenu(self._configMenu)
        self.tab().mainWindow().application().showZoomToolBar()

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
            _editorName = str(QFileDialog.getSaveFileName(self.tab(), "Choose editor", self._editorName, "Editor (*)", None , QFileDialog.DontConfirmOverwrite or QFileDialog.DontResolveSymlinks))
            if not os.path.exists(_editorName):
                _editorName = os.path.basename(_editorName)
        if _editorName != None and _editorName != "":
            self._editorName = _editorName
        self._saveIni()

    def save(self, filename=''):
        (config_path, fileName) = os.path.split(str(filename))
        configName = os.path.splitext(fileName)[0]
        if configName == self.dataAccessor().configName():
            self.plugin().application().errorMessage("Cannot overwrite original configuration file.")
        elif self.dataAccessor().isReplaceConfig():
            return TripleTabController.save(self, filename)
        elif filename != "":
            if TripleTabController.save(self, filename):
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
        if ini.has_option("config", "CurrentView"):
          proposed_view = ini.get("config", "CurrentView")
          if self._availableCenterViews.has_key(proposed_view):
            self._centerViewActions[self._currentCenterView].setChecked(False)
            self._currentCenterView = proposed_view
            self._centerViewActions[self._currentCenterView].setChecked(True)
        self._availableCenterViews[self._currentCenterView]()
        self.connect(self.tab().centerView(), SIGNAL("selected"), self.onSelected)
        if ini.has_option("config", "box content script"):
            self._configBrowserBoxView.setBoxContentScript(str(ini.get("config", "box content script")))
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
        ini.set("config", "CurrentView", self.currentCenterView())
        ini.set("config", "box content script", self._configBrowserBoxView.boxContentScript())
        self.plugin().application().writeIni()

    def exportDot(self, fileName=None):
        dot = DotExport()
        if self.currentCenterView() == "&Connection structure":
            presets = {'endpath':False, 'source':False, 'legend':False}
        else:
            presets = {'seqconnect':True, 'tagconnect':False, 'legend':False}
        for opt, val in presets.items():
            dot.setOption(opt, val)
        types = ""
        for ft in dot.file_types:
            if types != "":
                types += ";;"
            types += ft.upper() + " File (*." + ft.lower() + ")"
        filter = QString("PDF File (*.pdf)")
        if not fileName:
            defaultname = os.path.splitext(self._filename)[0] + "_export"
            fileName = str(QFileDialog.getSaveFileName(self.tab(), "Export dot graphic...", defaultname, types, filter))
        if fileName != "":
            name = fileName
            ext = str(filter).split(" ")[0].lower()
            if os.path.splitext(fileName)[1].lower().strip(".") in dot.file_types:
                name = os.path.splitext(fileName)[0]
                ext = os.path.splitext(fileName)[1].lower().strip(".")
            try:
                dot.export(self.dataAccessor(), name + "." + ext, ext)
            except Exception:
                self.plugin().application().errorMessage("Could not export dot graphic: " + exception_traceback())

    def setTab(self, tab):
        TripleTabController.setTab(self, tab)
        self._loadIni()

    def readFile(self, filename):
        """ Reads in the file in a separate thread.
        """
        thread = RunThread(self.dataAccessor().open, filename)
        while thread.isRunning():
            QCoreApplication.instance().processEvents()
        if thread.returnValue:
            if not self.dataAccessor().process() and self.dumpAction().isEnabled()==True:
                self.dumpAction().setEnabled(False)
                self.readOnlyAction().setChecked(True)
                self.readOnlyAction().setEnabled(False)
                self.readOnlyMode()
                self.plugin().application().warningMessage("Config does not contain a process and is opened in read-only mode.")
            if self.plugin().application().commandLineOptions().saveimage:
                self.tab().centerView().updateConnections()
                self.saveImage(self.plugin().application().commandLineOptions().saveimage)
                print "Saved image to", self.plugin().application().commandLineOptions().saveimage, "."
                sys.exit(2)
            return True
        return False
