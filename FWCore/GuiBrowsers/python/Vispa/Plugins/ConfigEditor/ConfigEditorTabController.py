import sys
import logging
import os.path
import copy

from PyQt4.QtCore import SIGNAL,QString,QCoreApplication
from PyQt4.QtGui import QMessageBox,QFileDialog

from Vispa.Main.Application import Application
from Vispa.Main.Exceptions import exception_traceback
from Vispa.Share.ThreadChain import ThreadChain
from Vispa.Plugins.Browser.BrowserTabController import BrowserTabController
from Vispa.Views.WidgetView import WidgetView
from Vispa.Plugins.ConfigEditor.ConfigEditorBoxView import ConfigEditorBoxView,ConnectionStructureView,SequenceStructureView
from Vispa.Gui.TextDialog import TextDialog

try:
    from FWCore.GuiBrowsers.DOTExport import DotExport
    import_dotexport_error=None
except Exception,e:
    import_dotexport_error=(str(e),exception_traceback())

try:
    from Vispa.Plugins.EdmBrowser.EventContentDialog import EventContentDialog
    event_content_error=None
except Exception,e:
    event_content_error=(str(e),exception_traceback())

try:
    from ToolDataAccessor import ToolDataAccessor,ConfigToolBase,standardConfigDir
    from ToolDialog import ToolDialog
    import_tools_error=None
except Exception,e:
    import_tools_error=(str(e),exception_traceback())

class ConfigEditorTabController(BrowserTabController):
    """
    """
    def __init__(self, plugin):
        logging.debug(__name__ + ": __init__")
        BrowserTabController.__init__(self, plugin)

        self._editorName = ""
        self._thread = None
        self._originalSizes=[100,1,200]
        self._toolDialog=None
        self._updateCenterView=False
        self.setEditable(False)
        
        self._configMenu = self.plugin().application().createPluginMenu('&Config')
        self._configToolBar = self.plugin().application().createPluginToolBar('&Config')
        openEditorAction = self.plugin().application().createAction('&Open in custom editor', self.openEditor, "F6")
        self._configMenu.addAction(openEditorAction)
        chooseEditorAction = self.plugin().application().createAction('&Choose custom editor...', self.chooseEditor, "Ctrl+T")
        self._configMenu.addAction(chooseEditorAction)
        self._configMenu.addSeparator()
        self._dumpAction = self.plugin().application().createAction('&Dump python config to single file...', self.dumpPython, "Ctrl+D")
        self._configMenu.addAction(self._dumpAction)
        self._dotExportAction = self.plugin().application().createAction('&Export dot graphic...', self.exportDot, "Ctrl+G")
        self._configMenu.addAction(self._dotExportAction)
        self._historyAction = self.plugin().application().createAction('&Show history...', self.history, "Ctrl+H")
        self._configMenu.addAction(self._historyAction)
        self._eventContentAction = self.plugin().application().createAction('&Browse event content...', self.eventContent, "Ctrl+Shift+C")
        self._configMenu.addAction(self._eventContentAction)
        self._configMenu.addSeparator()
        self._editorAction = self.plugin().application().createAction('&Edit using ConfigEditor', self.startEditMode, "F8")
        self._configMenu.addAction(self._editorAction)
        self._configToolBar.addAction(self._editorAction)
        
    #@staticmethod
    def staticSupportedFileTypes():
        """ Returns supported file type: py.
        """
        return [('py', 'Config file')]
    staticSupportedFileTypes = staticmethod(staticSupportedFileTypes)
    
    def dotExportAction(self):
        return self._dotExportAction
    
    def updateViewMenu(self):
        BrowserTabController.updateViewMenu(self)
        self.disconnect(self.tab().centerView(), SIGNAL("doubleClicked"), self.onCenterViewDoubleClicked)
        self.connect(self.tab().centerView(), SIGNAL("doubleClicked"), self.onCenterViewDoubleClicked)

    def onCenterViewDoubleClicked(self,object):
        logging.debug(__name__ + ": onCenterViewDoubleClicked()")
        self.tab().treeView().select(object)
        self.onTreeViewSelected(object)

    def updateCenterView(self, propertyView=True):
        """ Fill the center view from an item in the TreeView and update it """
        if not self._updateCenterView:
            # Do not update the very first time
            self._updateCenterView=True
            return
        statusMessage = self.plugin().application().startWorking("Updating center view")
        if propertyView:
            self.selectDataAccessor(True)
        else:
            self.selectDataAccessor(self.tab().propertyView().dataObject())
        if self._thread != None and self._thread.isRunning():
            self.dataAccessor().cancelOperations()
            while self._thread.isRunning():
                if not Application.NO_PROCESS_EVENTS:
                    QCoreApplication.instance().processEvents()
        objects = []
        select=self.tab().treeView().selection()
        if select != None:
            if self.currentCenterViewClassId() == self.plugin().viewClassId(ConnectionStructureView):
                self.tab().centerView().setArrangeUsingRelations(True)
                if self.tab().centerView().checkNumberOfObjects():
                    if self.dataAccessor().isContainer(select):
                        self._thread = ThreadChain(self.dataAccessor().readConnections, [select]+self.dataAccessor().allChildren(select))
                    else:
                        self._thread = ThreadChain(self.dataAccessor().readConnections, [select], True)
                    while self._thread.isRunning():
                        if not Application.NO_PROCESS_EVENTS:
                            QCoreApplication.instance().processEvents()
                    self.tab().centerView().setConnections(self._thread.returnValue())
                    self.tab().centerView().setDataObjects(self.dataAccessor().nonSequenceChildren(select))
                else:
                    self.tab().centerView().setDataObjects([])
            elif self.currentCenterViewClassId() == self.plugin().viewClassId(SequenceStructureView):
                self.tab().centerView().setArrangeUsingRelations(False)
                self.tab().centerView().setDataObjects([select])
                self.tab().centerView().setConnections({})
        if (self.currentCenterViewClassId() == self.plugin().viewClassId(ConnectionStructureView) or self.currentCenterViewClassId() == self.plugin().viewClassId(SequenceStructureView)) and \
            self.tab().centerView().updateContent(True):
            if not self.dataAccessor().isContainer(select) and self.currentCenterViewClassId() == self.plugin().viewClassId(ConnectionStructureView):
                    self.tab().centerView().select(select,500)
            else:
                self.tab().centerView().restoreSelection()
            select = self.tab().centerView().selection()
            if select != None:
                if self.tab().propertyView().dataObject() != select and propertyView:
                    self.tab().propertyView().setDataObject(select)
                    self.tab().propertyView().updateContent()
        if import_tools_error==None and self.tab().editorSplitter():
            self.updateConfigHighlight()
        self.plugin().application().stopWorking(statusMessage)
        
    def activated(self):
        """ Shows plugin menus when user selects tab.
        """
        logging.debug(__name__ + ": activated()")
        BrowserTabController.activated(self)
        self.plugin().application().showPluginMenu(self._configMenu)
        self.plugin().application().showPluginToolBar(self._configToolBar)
        self._editorAction.setVisible(not self.tab().editorSplitter())
        if self.tab().editorSplitter():
            self._applyPatToolAction.setVisible(self.dataAccessor().process()!=None)
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
        self.saveIni()

    def dumpPython(self, fileName=None):
        """ Dump python configuration to file """
        logging.debug(__name__ + ": dumpPython")
        dump = self.dataAccessor().dumpPython()
        if dump == None:
            logging.error(self.__class__.__name__ +": dumpPython() - "+"Cannot dump this config because it does not contain a 'process'.\nNote that only 'cfg' files contain a 'process'.")
            self.plugin().application().errorMessage("Cannot dump this config because it does not contain a 'process'.\nNote that only 'cfg' files contain a 'process'.")
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

    def history(self):
        """ Show config history """
        logging.debug(__name__ + ": history")
        history = self.dataAccessor().history()
        if history == None:
            logging.error(self.__class__.__name__ +": history() - "+"Cannot show config history because it does not contain a 'process'.\nNote that only 'cfg' files contain a 'process'.")
            self.plugin().application().errorMessage("Cannot show config history because it does not contain 'process'.\nNote that only 'cfg' files contain a 'process'.")
            return None
        dialog=TextDialog(self.tab(), "Configuration history", history, True, "This window lists the parameter changes and tools applied in this configuration file before it was loaded into ConfigEditor.")
        dialog.exec_()

    def eventContent(self):
        """ Open event content dialog """
        logging.debug(__name__ + ": eventContent")
        if event_content_error!=None:
            logging.error(__name__ + ": Could not import EventContentDialog: "+event_content_error[1])
            self.plugin().application().errorMessage("Could not import EventContentDialog (see logfile for details):\n"+event_content_error[0])
            return
        dialog=EventContentDialog(self.tab(),"This dialog let's you check if the input needed by your configuration file is in a dataformat or edm root file. You can compare either to a dataformat definition from a txt file (e.g. RECO_3_3_0) or any edm root file by selecting an input file.\n\nBranches that are used as input by your configuration but not present in the dataformat or file are marked in red.\nBranches that are newly created by your configuration are marked in green.")
        dialog.setConfigDataAccessor(self.dataAccessor())
        dialog.exec_()

    def loadIni(self):
        """ read options from ini """
        ini = self.plugin().application().ini()
        if ini.has_option("config", "editor"):
            self._editorName = str(ini.get("config", "editor"))
        else:
            self._editorName = "emacs"
        if ini.has_option("config", "CurrentView"):
            proposed_view = ini.get("config", "CurrentView")
        else:
            proposed_view = self.plugin().viewClassId(ConnectionStructureView)
        self.switchCenterView(proposed_view)
        if ini.has_option("config", "box content script") and isinstance(self.centerView(),ConfigEditorBoxView):
            self.centerView().setBoxContentScript(str(ini.get("config", "box content script")))
            self._boxContentDialog.setScript(str(ini.get("config", "box content script")))

    def scriptChanged(self, script):
        BrowserTabController.scriptChanged(self, script)
        self.saveIni()

    def saveIni(self):
        """ write options to ini """
        ini = self.plugin().application().ini()
        if not ini.has_section("config"):
            ini.add_section("config")
        ini.set("config", "editor", self._editorName)
        if self.currentCenterViewClassId():
            ini.set("config", "CurrentView", self.currentCenterViewClassId())
        if isinstance(self.centerView(),ConfigEditorBoxView):
            ini.set("config", "box content script", self.centerView().boxContentScript())
        self.plugin().application().writeIni()

    def exportDot(self, fileName=None):
        if import_dotexport_error!=None:
            logging.error(__name__ + ": Could not import DOTExport: "+import_dotexport_error[1])
            self.plugin().application().errorMessage("Could not import DOTExport (see logfile for details):\n"+import_dotexport_error[0])
            return
        dot = DotExport()
        if self.currentCenterViewClassId() == self.plugin().viewClassId(ConnectionStructureView):
            presets = {'seqconnect':False, 'tagconnect':True, 'seq':False, 'services':False, 'es':False, 'endpath':True, 'source':True, 'legend':False}
        else:
            presets = {'seqconnect':True, 'tagconnect':False, 'seq':True, 'services':False, 'es':False, 'endpath':True, 'source':True, 'legend':False}
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
                try:
                    dot.export(self.dataAccessor(), name + ".dot", "dot")
                    logging.error(self.__class__.__name__ +": exportDot() - "+"'dot' executable not found which is needed for conversion to '*." + ext + "'. Created '*.dot' file instead.")
                    self.plugin().application().errorMessage("'dot' executable not found which is needed for conversion to '*." + ext + "'. Created '*.dot' file instead.")
                except Exception,e:
                    logging.error(self.__class__.__name__ +": exportDot() - "+"Could not export dot graphic (see logfile for details): " + str(e))
                    self.plugin().application().errorMessage("Could not export dot graphic: " + exception_traceback())

    def readFile(self, filename):
        """ Reads in the file in a separate thread.
        """
        self._updateCenterView=False
        if self.dataAccessor().open(filename):
            self._dumpAction.setEnabled(self.dataAccessor().process()!=None)
            self._historyAction.setEnabled(self.dataAccessor().process()!=None)
            self._eventContentAction.setEnabled(self.dataAccessor().process()!=None)
            self._editorAction.setEnabled(self.dataAccessor().process()!=None)
            if self.plugin().application().commandLineOptions().saveimage:
                self.tab().centerView().updateConnections()
                self.saveImage(self.plugin().application().commandLineOptions().saveimage)
                print "Saved image to", self.plugin().application().commandLineOptions().saveimage, "."
                sys.exit(2)
            return True
        return False

    def save(self, filename=''):
        logging.debug(__name__ + ': save')
        self.startEditMode()
        if filename != "":
            if os.path.basename(filename) == os.path.basename(self.dataAccessor().configFile()):
                logging.error(self.__class__.__name__ +": save() - "+"Cannot use name of original configuration file: "+str(filename))
                self.plugin().application().errorMessage("Cannot use name of original configuration file.")
            elif BrowserTabController.save(self, filename):
                self.dataAccessor().setIsReplaceConfig()
                return True
            else:
                return False
        elif self.dataAccessor().isReplaceConfig():
            return BrowserTabController.save(self, filename)
        return self.tab().mainWindow().application().saveFileAsDialog()

    def writeFile(self, filename):
        """ Write replace config file.
        """
        logging.debug(__name__ + ': writeFile')
        
        text_file = open(filename, "w")
        text_file.write(self.toolDataAccessor().topLevelObjects()[0].dumpPython()[1])
        if self.dataAccessor().process():
            text_file.write(self.dataAccessor().process().dumpHistory(False))
        text_file.close()
        return True

    def open(self, filename=None, update=True):
        if BrowserTabController.open(self, filename, update):
            if self.dataAccessor().isReplaceConfig():
                self.startEditMode()
            return True
        return False

    def startEditMode(self):
        logging.debug(__name__ + ": startEditMode")
        if import_tools_error!=None:
            logging.error(__name__ + ": Could not import tools for ConfigEditor: "+import_tools_error[1])
            self.plugin().application().errorMessage("Could not import tools for ConfigEditor (see logfile for details):\n"+import_tools_error[0])
            return
        if self.tab().editorSplitter():
            return
        if self._filename and not self.dataAccessor().process():
            logging.error(__name__ + ": Config does not contain a process and cannot be edited using ConfigEditor.")
            self.plugin().application().errorMessage("Config does not contain a process and cannot be edited using ConfigEditor.")
            return
        if self._filename and not self.dataAccessor().isReplaceConfig():
            self.setFilename(None)
            self.updateLabel()
        self.tab().createEditor()
        self.setEditable(True)
        self.tab().verticalSplitter().setSizes(self._originalSizes)

        self._importAction = self.plugin().application().createAction('&Import configuration...', self.importButtonClicked, "F2")
        self._configMenu.addAction(self._importAction)
        self._configToolBar.addAction(self._importAction)
        self._applyPatToolAction = self.plugin().application().createAction('&Apply tool...', self.applyButtonClicked, "F3")
        self._configMenu.addAction(self._applyPatToolAction)
        self._configToolBar.addAction(self._applyPatToolAction)
        self.activated()

        self._toolDataAccessor=ToolDataAccessor()
        self._toolDataAccessor.setConfigDataAccessor(self.dataAccessor())
        self.tab().editorTableView().setDataAccessor(self._toolDataAccessor)
        self.connect(self.tab().editorTableView(), SIGNAL('importButtonClicked'), self.importButtonClicked)
        self.connect(self.tab().editorTableView(), SIGNAL('applyButtonClicked'), self.applyButtonClicked)
        self.connect(self.tab().editorTableView(), SIGNAL('removeButtonClicked'), self.removeButtonClicked)
        self.connect(self.tab().editorTableView(), SIGNAL('selected'), self.codeSelected)
        self.connect(self.tab().propertyView(), SIGNAL('valueChanged'), self.valueChanged)
        self._updateCode()

    def toolDataAccessor(self):
        return self._toolDataAccessor

    def minimizeEditor(self):
        if self.tab().originalButton().isChecked():
            self._originalSizes=self.tab().verticalSplitter().sizes()
        self.tab().minimizeButton().setChecked(True)
        self.tab().originalButton().setChecked(False)
        self.tab().maximizeButton().setChecked(False)
        self.tab().verticalSplitter().setSizes([100, 1, 0])
    
    def originalEditor(self):
        self.tab().minimizeButton().setChecked(False)
        self.tab().originalButton().setChecked(True)
        self.tab().maximizeButton().setChecked(False)
        self.tab().verticalSplitter().setSizes(self._originalSizes)

    def maximizeEditor(self):
        if self.tab().originalButton().isChecked():
            self._originalSizes=self.tab().verticalSplitter().sizes()
        self.tab().minimizeButton().setChecked(False)
        self.tab().originalButton().setChecked(False)
        self.tab().maximizeButton().setChecked(True)
        self.tab().verticalSplitter().setSizes([0, 1, 100])
    
    def _updateCode(self,propertyView=True):
        logging.debug(__name__ + ": _updateCode")
        self.tab().propertyView().setEnabled(False)
        self.toolDataAccessor().updateToolList()
        self.tab().editorTableView().setDataObjects(self.toolDataAccessor().topLevelObjects())
        if self.tab().editorTableView().updateContent():
            self.tab().editorTableView().restoreSelection()
        self.updateContent(False,propertyView)
        self.tab().propertyView().setEnabled(True)

    def importConfig(self,filename):
        logging.debug(__name__ + ": importConfig")
        statusMessage = self.plugin().application().startWorking("Import python configuration in Editor")
        try:
            good=self.open(filename,False)
        except:
            logging.error(__name__ + ": Could not open configuration file: "+exception_traceback())
            self.plugin().application().errorMessage("Could not open configuration file (see log file for details).")
            self.plugin().application().stopWorking(statusMessage,"failed")
            return False
        if not good:
            logging.error(__name__ + ": Could not open configuration file.")
            self.plugin().application().errorMessage("Could not open configuration file.")
            self.plugin().application().stopWorking(statusMessage,"failed")
            return False
        if not self.dataAccessor().process():
            logging.error(__name__ + ": Config does not contain a process and cannot be edited using ConfigEditor.")
            self.plugin().application().errorMessage("Config does not contain a process and cannot be edited using ConfigEditor.")
            self.plugin().application().stopWorking(statusMessage,"failed")
            return False
        if self._filename and not self.dataAccessor().isReplaceConfig():
            self.setFilename(None)
            self.updateLabel()
        self.toolDataAccessor().setConfigDataAccessor(self.dataAccessor())
        self.tab().propertyView().setDataObject(None)
        self._updateCode()
        self._applyPatToolAction.setVisible(True)
        self.plugin().application().stopWorking(statusMessage)
        return True

    def updateConfigHighlight(self):
        if self.tab().editorTableView().selection() in self.toolDataAccessor().toolModules().keys():
            self.tab().centerView().highlight(self.toolDataAccessor().toolModules()[self.tab().editorTableView().selection()])
        else:
            self.tab().centerView().highlight([])
            
    def importButtonClicked(self):
        logging.debug(__name__ + ": importButtonClicked")
        filename = QFileDialog.getOpenFileName(
            self.tab(),'Select a configuration file',standardConfigDir,"Python configuration (*.py)")
        if not filename.isEmpty():
            self.importConfig(str(filename))

    def applyButtonClicked(self):
        logging.debug(__name__ + ": applyButtonClicked")
        if not self._toolDialog:
            self._toolDialog=ToolDialog()
        self._toolDialog.setDataAccessor(self._toolDataAccessor)
        if not self._toolDialog.exec_():
            return
        if not self.toolDataAccessor().addTool(self._toolDialog.tool()):
            return
        self.setModified(True)
        self._updateCode()
        self.tab().editorTableView().select(self.tab().editorTableView().dataObjects()[-2])
        self.codeSelected(self.tab().editorTableView().dataObjects()[-2])
            
    def removeButtonClicked(self,object):
        logging.debug(__name__ + ": removeButtonClicked")
        if not object or not self.dataAccessor().process() or\
            self._toolDataAccessor.label(object) in ["Import","ApplyTool"]:
            return
        if not self.toolDataAccessor().removeTool(object):
            self.plugin().application().errorMessage("Could not apply tool. See log file for details.")
            return
        self.setModified(True)
        self._updateCode()
        self.tab().editorTableView().select(self.tab().editorTableView().dataObjects()[-1])
        self.codeSelected(self.tab().editorTableView().dataObjects()[-1])

    def onSelected(self, select):
        self.selectDataAccessor(select)
        BrowserTabController.onSelected(self, select)

    def refresh(self):
        self.tab().propertyView().setDataObject(None)
        BrowserTabController.refresh(self)

    def updateContent(self, filtered=False, propertyView=True):
        if import_tools_error==None and isinstance(object,ConfigToolBase):
            propertyView=False
        else:
            self.tab().propertyView().setDataAccessor(self.dataAccessor())
        BrowserTabController.updateContent(self, filtered, propertyView)

    def select(self, object):
        self.selectDataAccessor(object)
        BrowserTabController.select(self, object)
    
    def selectDataAccessor(self,object):
        if import_tools_error==None and isinstance(object,ConfigToolBase):
            self.tab().propertyView().setDataAccessor(self.toolDataAccessor())
        else:
            self.tab().propertyView().setDataAccessor(self.dataAccessor())
    
    def codeSelected(self,select):
        if self.tab().propertyView().dataObject() != select:
            statusMessage = self.plugin().application().startWorking("Updating property view")
            self.tab().propertyView().setDataAccessor(self.toolDataAccessor())
            self.tab().propertyView().setDataObject(select)
            self.tab().propertyView().updateContent()
            self.plugin().application().stopWorking(statusMessage)
        self.updateConfigHighlight()

    def valueChanged(self,name):
        if isinstance(self.tab().propertyView().dataObject(),ConfigToolBase):
            if self._toolDataAccessor.label(self.tab().propertyView().dataObject())=="Import":
                filename=self.toolDataAccessor().propertyValue(self.tab().propertyView().dataObject(),"filename")
                return self.importConfig(filename)
            else:
                self.toolDataAccessor().updateProcess()
                self.setModified(True)
                self._updateCode(False)
                self.codeSelected(self.tab().editorTableView().selection())
        else:
            self._updateCode()
