import os.path

from ConfigReader import *
from TreeView import *
from ModuleView import *
from Tab import *
from Icons import *

from qt import *

class ConfigBrowserTab(Tab):
    def __init__(self, parent,name,filename=""):
        """ constructor """
        Tab.__init__(self,parent,name,filename)

        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserTab.__init__"]
        
        self.fillMenu()
        self._editorName=""
        self.readIni()
        
        self._box_properties = [("Label","","label"),
                               ("Type","","getType"),
                               ("Classname","","getClassname"),
                               ("Filename","","getFilename"),
                               ("Package","","getPackage")]

        self.tree_view = TreeView(self,"tree_view")
        self.box_view = ModuleView(self,"box_view")
        self.setSizes([self.tree_view.width(),self.contentsRect().width()-self.handleWidth()-self.tree_view.width()])

        self.connect(self,PYSIGNAL("objectsSelected()"),self.updateConnections)
        self.connect(self,PYSIGNAL("tabVisible()"),self.showMenu)
        self.connect(self,PYSIGNAL("tabInvisible()"),self.hideMenu)

        self._file=ConfigReader(filename,self.parentwindow)
        self.objects=self._file.objects
        self.connections=self._file.connections
        self.dumpPythonAction.setVisible(hasattr(self._file.config,"process"))

    def fillMenu(self):
        """ Fill entries in menu """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserTab.fillMenu"]
        
        self.ConfigMenu = QPopupMenu(self.parentwindow)
        self.ConfigMenu.id=self.parentwindow.MenuBar.count()+1
        self.parentwindow.MenuBar.insertItem(QString(""),self.ConfigMenu,self.ConfigMenu.id)
        self.parentwindow.MenuBar.findItem(self.ConfigMenu.id).setText("&Config")

        self.refreshAction = QAction("&Refresh config file","F5",self.parentwindow,"refreshAction")
        self.connect(self.refreshAction,SIGNAL("activated()"),self.refreshFile)

        self.image1 = QPixmap()
        self.image1.loadFromData(exit_image,"PNG")
        self.openEditorAction = QAction(QIconSet(self.image1),"&Open editor...","F6",self.parentwindow,"openEditorAction")
        self.connect(self.openEditorAction,SIGNAL("activated()"),self.openEditor)
        
        self.chooseEditorAction = QAction("Choose &editor...","",self.parentwindow,"chooseEditorAction")
        self.connect(self.chooseEditorAction,SIGNAL("activated()"),self.chooseEditor)

        self.dumpPythonAction = QAction("&Dump python config to single file...","",self.parentwindow,"dumpPythonAction")
        self.connect(self.dumpPythonAction,SIGNAL("activated()"),self.dumpPython)

        self.toolBar=QToolBar(QString("Config"),self.parentwindow,Qt.DockTop)
        self.openEditorAction.addTo(self.toolBar)
        
        self.refreshAction.addTo(self.ConfigMenu)
        self.dumpPythonAction.addTo(self.ConfigMenu)
        self.ConfigMenu.insertSeparator()
        self.openEditorAction.addTo(self.ConfigMenu)
        self.chooseEditorAction.addTo(self.ConfigMenu)

    def showMenu(self):
        """ show entries in MainMenu """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserTab.showMenu"]
        
        self.parentwindow.MenuBar.setItemVisible(self.ConfigMenu.id,True)
        self.chooseEditorAction.setVisible(True)
        self.openEditorAction.setVisible(True)
        self.refreshAction.setVisible(True)
        self.dumpPythonAction.setVisible(hasattr(self._file.config,"process"))
        self.toolBar.show()

    def hideMenu(self):
        """ hide entries in MainMenu """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserTab.hideMenu"]
        
        self.parentwindow.MenuBar.setItemVisible(self.ConfigMenu.id,False)
        self.chooseEditorAction.setVisible(False)
        self.openEditorAction.setVisible(False)
        self.refreshAction.setVisible(False)
        self.dumpPythonAction.setVisible(False)
        self.toolBar.hide()
        
    def updateConnections(self):
        """ when objects are selected """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserTab.updateConnections"]
        
        self._file.readConnections(self.selected_objects)
        
    def refreshFile(self):
        """ reload config file """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserTab.refreshFile"]
        
        self.parentwindow.saveWindowState()
        self._file.openFile()
        self.objects=self._file.objects
        self.connections=self._file.connections
        self.parentwindow.emit(PYSIGNAL("refreshObjects()"),(False,))
        self.parentwindow.restoreWindowState()

    def openEditor(self):
        """ Call editor """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserTab.openEditor"]
        
        selected_object=self.parentwindow.selected_object
        if self._editorName!="" and selected_object!=None and os.path.exists(selected_object.getFullFilename()):
            if os.path.expandvars("$CMSSW_RELEASE_BASE") in selected_object.getFullFilename():
                QMessageBox.information(self,"Opening readonly file...","This file is from $CMSSW_RELEASE_BASE and readonly",0) 
            command=self._editorName
            command+=" "+selected_object.getFullFilename()
            command+=" &"
            os.system(command)

    def chooseEditor(self,_editorName=""):
        """ Choose editor using FileDialog """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserTab.chooseEditor"]
        
        if _editorName=="":
            _editorName = str(QFileDialog.getSaveFileName(".","Editor (*)",self,"open editor dialog","Choose editor" ))
            if not os.path.exists(_editorName):
                _editorName=os.path.basename(_editorName)
        if _editorName!=None and _editorName!="":
            self._editorName=_editorName
        self.writeIni()

    def dumpPython(self,fileName=None):
        """ Choose editor using FileDialog """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserTab.chooseEditor"]
        
        filter=QString("")
        while fileName==None or (os.path.exists(fileName) and not QMessageBox.question(self,"Overwrite file","File already exists. Do you want to overwrite?",QMessageBox.Yes,QMessageBox.Cancel)==QMessageBox.Yes):
            fileName = str(QFileDialog.getSaveFileName(".","Python config (*.py)",self,"save python config","Save python config...",filter))
        if fileName!="":
            self.parentwindow.StatusBar.message("saving python config...")
            name=fileName
            ext="PY"
            if os.path.splitext(fileName)[1].upper().strip(".")==ext:
                name=os.path.splitext(fileName)[0]
                ext=os.path.splitext(fileName)[1].upper().strip(".")
            text_file = open(name+"."+ext.lower(), "w")
            text_file.write(self._file.dumpPython())
            text_file.close()
            self.parentwindow.StatusBar.message("saving python config...done")

    def readIni(self):
        """ read options from ini """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserTab.readIni"]
        
        ini=self.parentwindow.loadIni()
        if ini.has_option("editor", "filename"):
            self._editorName=str(ini.get("editor", "filename"))

    def writeIni(self):
        """ write options to ini """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserTab.writeIni"]
        
        ini=self.parentwindow.loadIni()
        if not ini.has_section("editor"):
            ini.add_section("editor")
        ini.set("editor", "filename",self._editorName)
        self.parentwindow.saveIni(ini)
