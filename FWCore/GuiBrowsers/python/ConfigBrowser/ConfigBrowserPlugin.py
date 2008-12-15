import os.path

from ConfigBrowserTab import *
from Icons import *

from qt import *

class ConfigBrowserPlugin(QWidget):
    """ MainWindow of ConfigBrowser """
    def __init__(self, parent=None, name=None, fl=0):
        """ constructor """
        self.parentwindow=parent

        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserPlugin.__init__"]
        
        QWidget.__init__(self,parent,name)
        
        self._includePaths=[]
        
        self.readIni()
        i=0
        for path in self._includePaths:
            sys.path.insert(i,path)
            i+=1
        
        self.fillMenu()

        self.connect(self.parentwindow,PYSIGNAL("openFile()"),self.openConfigFile)

    def fillMenu(self):
        """ fill entries in MainMenu """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserPlugin.fillMenu"]
        
        self.openFileAction = QAction("&Open config file...","Ctrl+O",self.parentwindow,"openFileAction")
        self.image0 = QPixmap()
        self.image0.loadFromData(open_image,"PNG")
        self.openFileAction.setIconSet(QIconSet(self.image0))
        self.connect(self.openFileAction,SIGNAL("activated()"),self.openConfigFile)

        self.openFileAction.addTo(self.parentwindow.toolBar)

        self.parentwindow.fileMenuItems=[self.openFileAction,0]+self.parentwindow.fileMenuItems
        self.parentwindow.updateMenu()

    def openConfigFile(self,filename=""):
        """ Call ConfigReader and show file in TreeView """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserPlugin.openConfigFile"]
        
        if filename=="":
            filename = str(QFileDialog.getOpenFileName(".","Python config (*.py)",self,"open file dialog","Choose a file to open" ))
        fileName = os.path.basename(str(filename))
        ext=os.path.splitext(fileName)[1].lower().strip(".")
        if os.path.exists(filename) and ext=="py":
            self.parentwindow.StatusBar.message("opening config file "+fileName+"...")
            shortname=fileName
            if len(os.path.splitext(fileName)[0])>20:
                shortname=os.path.splitext(fileName)[0][0:20]+"...."+ext
            tab=ConfigBrowserTab(self.parentwindow.tabWidget,shortname,filename)
            self.parentwindow.tabWidget.showPage(tab)
            self.parentwindow.emit(PYSIGNAL("fileOpened()"),(filename,))
            self.parentwindow.emit(PYSIGNAL("refreshObjects()"),())
            self.parentwindow.StatusBar.message("opening config file "+fileName+"...done")
    
    def readIni(self):
        """ read options from ini """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserPlugin.readIni"]
        
        ini=self.parentwindow.loadIni()
        if self._includePaths==[]:
            if ini.has_option("history", "includePaths"):
                text=str(ini.get("history", "includePaths"))
                self._includePaths=text.strip("[']").replace("', '",",").split(",")
                if self._includePaths==[""]:
                    self._includePaths=[]
        
    def writeIni(self):
        """ write options to ini """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ConfigBrowserPlugin.writeIni"]
      
        ini=self.parentwindow.loadIni()
        if not ini.has_section("history"):
            ini.add_section("history")
        ini.set("history", "includePaths",self._includePaths)
        self.parentwindow.saveIni(ini)
