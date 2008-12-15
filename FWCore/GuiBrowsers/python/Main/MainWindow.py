#! /usr/bin/env python
import sys
import os.path
import ConfigParser

from PropertiesView import *
from Icons import *

from qt import *

class MainWindow(QMainWindow):
    """ MainWindow """
    def __init__(self, parent=None, name=None, fl=0):
        """ constructor creates views """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.__init__"]
        
        QMainWindow.__init__(self,parent,name,fl)
        self.clearWState(Qt.WState_Polished)
        self.setCaption("MainWindow")

        self.MenuBar=self.menuBar()
        self.StatusBar=self.statusBar()

        self._lastFiles=[]
        
        (dirName, fileName) = os.path.split(sys.argv[0])
        self._iniFileName=""
        if dirName!="":
            self._iniFileName+=dirName+"/"
        self._iniFileName+=os.path.splitext(fileName)[0]+".ini"

        self.readIni()

        self.fileMenuItems=[]
        self.fillMenu()
        self.updateMenu()

        self.right_splitter=QSplitter(self)
        self.setCentralWidget(self.right_splitter)
        self.right_splitter.adjustSize()
        self.right_splitter.resize(self.childrenRect().size())
        self.tabWidget=QTabWidget(self.right_splitter,"Tabs")
        self.tabWidget.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding,0)
        self.properties_view = PropertiesView(self.right_splitter,"properties_view")
        self.right_splitter.setSizes([self.right_splitter.contentsRect().width()-self.right_splitter.handleWidth()-self.properties_view.sizeHint().width(),self.properties_view.sizeHint().width()])
        
        self.selected_object=None
        self._found=[]
        self._foundIndex=0
        self._saved_selected_object_index=0
        self._saved_selected_object_label=""
        self._saved_selected_objects_index=0
        self._saved_selected_objects_label=""

        self.connect(self,PYSIGNAL("clearObjects()"),self.clearObjects)
        self.connect(self,PYSIGNAL("selectObject()"),self.selectObject)
        self.connect(self,PYSIGNAL("selectObjects()"),self.selectObjects)
        self.connect(self,PYSIGNAL("refreshObjects()"),self.refreshObjects)
        self.connect(self,PYSIGNAL("fileOpened()"),self.fileOpened)
        self.connect(self.tabWidget,SIGNAL("currentChanged(QWidget *)"),self.selectTab)
        self.connect(self.MenuBar,SIGNAL("activated(int)"),self.menuActivated)

    def selectObject(self,object=None,byClick=True):
        """ Show Configobject in PropertiesView """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.selectObject"]
        
        if hasattr(object,"obj"):
            object=object.obj

        self.tabWidget.currentPage().emit(PYSIGNAL("selectObject()"),(object,byClick))
        self.selected_object=object
        self.tabWidget.currentPage().emit(PYSIGNAL("objectSelected()"),(byClick,))
        self.emit(PYSIGNAL("objectSelected()"),(byClick,))

    def getObjectsRecursive(self,entry,objects):
        """ Get objects daughters and siblings recursively """
        if entry!=None:
            objects+=[entry]
            for daughter in entry.daughters:
                if not daughter in objects:
                    self.getObjectsRecursive(daughter,objects)
        return objects

    def selectObjects(self,object=None,update=True):
        """ Show Object in BoxView and PropertiesView """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.selectObjects"]
        
        if hasattr(object,"obj"):
            object=object.obj

        self.tabWidget.currentPage().emit(PYSIGNAL("selectObjects()"),(object,))
        self.tabWidget.currentPage().selected_object=object
        self.tabWidget.currentPage().selected_objects=self.getObjectsRecursive(object,[])
        self.tabWidget.currentPage().emit(PYSIGNAL("objectsSelected()"),())
            
        if update:
            so=None
            if self.tabWidget.currentPage().selected_objects!=[]:
                so=self.tabWidget.currentPage().selected_objects[0]
            self.emit(PYSIGNAL("selectObject()"),(so,False))

    def refreshObjects(self,update=True):
        """ Refresh all Objects """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.refreshObjects"]

        self.emit(PYSIGNAL("clearObjects()"),())
        self.tabWidget.currentPage().emit(PYSIGNAL("refreshObjects()"),(update,))
        self.showFindMenu()

    def selectTab(self,tab):
        """ when a different tab is selected update menu """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.selectTab"]

        self.updateCaption()
        self.emit(PYSIGNAL("clearObjects()"),())
        for i in range(self.tabWidget.count()):
            if i==self.tabWidget.currentPageIndex():
                self.tabWidget.page(i).emit(PYSIGNAL("tabVisible()"),())
            else:
                self.tabWidget.page(i).emit(PYSIGNAL("tabInvisible()"),())
        self.showFindMenu()

    def updateCaption(self):
        """ update window caption """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.updateCaption"]
        
        name=""
        if hasattr(self.tabWidget.currentPage(),"filename"):
            filename=self.tabWidget.currentPage().filename
            dirName = os.path.dirname(sys.argv[0])
            if os.path.abspath(dirName) in filename:
                filename=filename[len(os.path.abspath(dirName))+1:]
            name=" - "+filename
        self.setCaption(str(self.caption()).split("-")[0].strip()+name)

    def fillMenu(self):
        """ fill entries in MainMenu """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.fillMenu"]
        
        self.fileMenu = QPopupMenu(self)
        self.fileMenu.id=self.MenuBar.count()+1
        self.MenuBar.insertItem(QString(""),self.fileMenu,self.fileMenu.id)
        self.MenuBar.findItem(self.fileMenu.id).setText("&File")

        self.fileExitAction = QAction("E&xit","",self,"fileExitAction")
        self.connect(self.fileExitAction,SIGNAL("activated()"),self.close)

        self.image0 = QPixmap()
        self.image0.loadFromData(close_image,"PNG")
        self.fileCloseAction = QAction(QIconSet(self.image0),"&Close","Ctrl+W",self,"fileCloseAction")
        self.connect(self.fileCloseAction,SIGNAL("activated()"),self.closeTab)

        self.aboutAction = QAction("&About","F1",self,"aboutAction")
#        self.image1 = QPixmap(vispa_image)
#        self.aboutAction.setIconSet(QIconSet(self.image1))
        self.connect(self.aboutAction,SIGNAL("activated()"),self.about)

        self.toolBar=QToolBar(QString("File"),self,Qt.DockTop)
#        self.aboutAction.addTo(self.toolBar)
        self.fileCloseAction.addTo(self.toolBar)

        self.fileMenuItems+=[self.fileCloseAction,0,self.aboutAction,0,self.fileExitAction]

        self.Find = QPopupMenu(self)
        self.Find.id=self.MenuBar.count()+1
        self.MenuBar.insertItem(QString(""),self.Find,self.Find.id)
        self.MenuBar.findItem(self.Find.id).setText("F&ind")

        self.image2 = QPixmap()
        self.image2.loadFromData(find_label_image,"PNG")
        self.findByLabelAction = QAction(QIconSet(self.image2),"&Find by label...","Ctrl+F",self,"findByLabelAction")
        self.connect(self.findByLabelAction,SIGNAL("activated()"),self.findByLabel)
    
        self.image5 = QPixmap()
        self.image5.loadFromData(find_property_image,"PNG")
        self.findByPropertyAction = QAction(QIconSet(self.image5),"F&ind by property...","Ctrl+Shift+F",self,"findByPropertyAction")
        self.connect(self.findByPropertyAction,SIGNAL("activated()"),self.findByProperty)
    
        self.image4 = QPixmap()
        self.image4.loadFromData(findprevious_image,"PNG")
        self.findPreviousAction = QAction(QIconSet(self.image4),"Find &previous","Ctrl+P",self,"findPreviousAction")
        self.connect(self.findPreviousAction,SIGNAL("activated()"),self.findPrevious)

        self.image3 = QPixmap()
        self.image3.loadFromData(findnext_image,"PNG")
        self.findNextAction = QAction(QIconSet(self.image3),"Find &next","Ctrl+N",self,"findNextAction")
        self.connect(self.findNextAction,SIGNAL("activated()"),self.findNext)
    
        self.findByLabelAction.addTo(self.Find)
        self.findByPropertyAction.addTo(self.Find)
        self.findPreviousAction.addTo(self.Find)
        self.findNextAction.addTo(self.Find)

        self.findToolBar=QToolBar(QString("Find"),self,Qt.DockTop)
        self.findByLabelAction.addTo(self.findToolBar)
        self.findByPropertyAction.addTo(self.findToolBar)
        self.findPreviousAction.addTo(self.findToolBar)
        self.findIndex=QLabel(QString(""),self.findToolBar)
        self.findNextAction.addTo(self.findToolBar)

        self.MenuBar.setItemVisible(self.Find.id,False)
        self.findToolBar.hide()
    
    def showFindMenu(self):
        """ shows the find menu and creates it if necessary """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.showFindMenu"]

        self.MenuBar.setItemVisible(self.Find.id,True)
        self.findByLabelAction.setVisible(True)
        self.findByPropertyAction.setVisible(True)
        self.findToolBar.show()

    def hideFindMenu(self):
        """ hide find entries in MainMenu """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.hideFindMenu"]
        
        self.MenuBar.setItemVisible(self.Find.id,False)
        self.findByLabelAction.setVisible(False)
        self.findByPropertyAction.setVisible(False)
        self.findPreviousAction.setVisible(False)
        self.findIndex.setText("")
        self.findIndex.hide()
        self.findNextAction.setVisible(False)
        self.findToolBar.hide()

    def menuActivated(self,id):
        """ when last opened file is clicked in menu """
        if self.fileMenu.indexOf(id)>len(self.fileMenuItems)-4 and self.fileMenu.indexOf(id)<len(self.fileMenuItems)-3+len(self._lastFiles):
            self.emit(PYSIGNAL("openFile()"),(self._lastFiles[self.fileMenu.indexOf(id)-(len(self.fileMenuItems)-3)],))

    def updateMenu(self):
        """ update entries in MainMenu """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.updateMenu"]
        
        fileItems=[]
        for filename in self._lastFiles:
            fileName = os.path.basename(str(filename))
            setattr(self,fileName,QAction(self,fileName))
            getattr(self,fileName).setText(fileName)
            getattr(self,fileName).setMenuText(fileName)
            getattr(self,fileName).isFilename=True
            fileItems+=[getattr(self,fileName)]
        if fileItems!=[]:
            fileItems+=[0]
        MenuItems=self.fileMenuItems[:-3]+fileItems+self.fileMenuItems[-3:]
        self.fileMenu.clear()
        for item in MenuItems:
            if item==0:
                self.fileMenu.insertSeparator()
            else:
                item.addTo(self.fileMenu)

    def closeTab(self):
        """ close current tab """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.closeTab"]
        
        self.emit(PYSIGNAL("clearObjects()"),())
        if self.tabWidget.currentPage()!=None:
            tab=self.tabWidget.currentPage()
            tab.emit(PYSIGNAL("clearObjects()"),())
            if self.tabWidget.count()==1:
                tab.emit(PYSIGNAL("tabInvisible()"),())
            self.tabWidget.removePage(tab)
            tab.close()
            del tab
            if self.tabWidget.count()==0:
                self.updateCaption()
        else:
            self.close()

    def fileOpened(self,filename):
        """ add filename to list of last opened files """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.fileOpened"]
        
        if filename in self._lastFiles:
            del self._lastFiles[self._lastFiles.index(filename)]
        self._lastFiles=[filename]+self._lastFiles[:7]
        self.writeIni()
        self.updateMenu()
        self.updateCaption()
 
    def clearObjects(self):
        """ Clear object """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.clearObjects"]
        
        self.selected_object=None
        self._found=[]
        self._foundIndex=0
        self.hideFindMenu()

    def about(self):
        """ Show about Message """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.about"]
        
        QMessageBox.information(self,"About this software...",str(self.caption()).split("-")[0].strip()+" - "+self.version,0) 

    def findLabel(self,name,excl=False):
        """ find name in objects """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.findLabel"]

        found=[]
        if self.tabWidget.currentPage()!=None:
            for o in self.tabWidget.currentPage().objects:
                if (excl and name==o.label) or (not excl and name.strip().lower() in o.label.lower()):
                    found+=[o]
        return found
    
    def findProperty(self,property):
        """ find name in objects """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.findProperty"]

        prop=property.split("=")
        name=""
        value=""
        if len(prop)>0:
            name=prop[0]
        if len(prop)>1:
            value=prop[1]
        
        found=[]
        if self.tabWidget.currentPage()!=None:
            for o in self.tabWidget.currentPage().objects:
                hasProperty=False
                for property in o.properties:
                    if name.strip().lower() in property[1].lower() and value.strip().lower() in property[2].lower():
                        hasProperty=True
                if hasProperty:
                    found+=[o]
        return found
    
    def findByLabel(self):
        """ Find object by label """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.findByLabel"]
            
        name = str(QInputDialog.getText("Find object...","Enter phrase to find in object label (not case sensitive):")[0])
        if name!="":
            self._found=self.findLabel(name)
            self._foundIndex=-1
            self.findPreviousAction.setVisible(True)
            self.findIndex.show()
            self.findNextAction.setVisible(True)
            self.findNext()

    def findByProperty(self):
        """ Find object by property """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.findByProperty"]
            
        name = str(QInputDialog.getText("Find object...","Enter phrase to find in object properties (not case sensitive). Separate name and value by \'=\', e.g. \'package=PatAlgos\')")[0])
        if name!="":
            self._found=self.findProperty(name)
            self._foundIndex=-1
            self.findPreviousAction.setVisible(True)
            self.findIndex.show()
            self.findNextAction.setVisible(True)
            self.findNext()

    def findNext(self):
        """ Select next found object in tree """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.findNext"]
            
        if self._foundIndex<len(self._found)-1:
            self._foundIndex+=1
            self.emit(PYSIGNAL("selectObjects()"),(self._found[self._foundIndex],))
        self.findIndex.setText(str(self._foundIndex+1)+"/"+str(len(self._found)))

    def findPrevious(self):
        """ Select previous found object in tree """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.findPrevious"]
            
        if self._foundIndex>0:
            self._foundIndex-=1
            self.emit(PYSIGNAL("selectObjects()"),(self._found[self._foundIndex],))
        self.findIndex.setText(str(self._foundIndex+1)+"/"+str(len(self._found)))

    def saveSelectedObject(self):
        """ save selected objects """
        if self.selected_object!=None:
            so=self.selected_object
            self._saved_selected_object_label=so.label
            found_object=self.findLabel(self._saved_selected_object_label,True)
            for o in found_object:
                if o==so:
                    self._saved_selected_object_index=found_object.index(o)
        
    def saveSelectedObjects(self):
        """ save selected objects """
        if self.tabWidget.currentPage().selected_object!=None:
            so=self.tabWidget.currentPage().selected_object
            self._saved_selected_objects_label=so.label
            found_objects=self.findLabel(self._saved_selected_objects_label,True)
            for o in found_objects:
                if o==so:
                    self._saved_selected_objects_index=found_objects.index(o)
                    
    def restoreSelectedObject(self):
        """ restore selected object """
        found_object=self.findLabel(self._saved_selected_object_label,True)
        new_selected_object=None
        if len(found_object)>self._saved_selected_object_index:
            new_selected_object=found_object[self._saved_selected_object_index]
        elif len(found_object)>0:
            new_selected_object=found_object[0]

        self.emit(PYSIGNAL("selectObject()"),(new_selected_object,False))

    def restoreSelectedObjects(self,update=True):
        """ restore selected objects """
        found_objects=self.findLabel(self._saved_selected_objects_label,True)
        new_selected_objects=None
        if len(found_objects)>self._saved_selected_objects_index:
            new_selected_objects=found_objects[self._saved_selected_objects_index]
        elif len(found_objects)>0:
            new_selected_objects=found_objects[0]
        elif len(self.tabWidget.currentPage().objects)>0:
            new_selected_objects=self.tabWidget.currentPage().objects[0]

        self.emit(PYSIGNAL("selectObjects()"),(new_selected_objects,update))

    def saveWindowState(self):
        """ save the current state of the window content """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.saveWindowState"]
            
        self.saveSelectedObjects()
        self.saveSelectedObject()
        
    def restoreWindowState(self):
        """ restore the saved state of the window content """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.restoreWindowState"]
            
        self.restoreSelectedObjects(False)
        self.restoreSelectedObject()
        
    def resizeEvent(self,e):
        """ save size to ini on resize window """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.resizeEvent"]
            
        QMainWindow.resizeEvent(self,e)
        self.writeIni()

    def loadIni(self):
        """ read ini """
        ini=ConfigParser.ConfigParser()
        ini.read(self._iniFileName)
        return ini

    def saveIni(self,ini):
        """ write ini """
        configfile=open(self._iniFileName,"w")
        ini.write(configfile)

    def readIni(self):
        """ read options from ini """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.readIni"]
        
        ini=self.loadIni()
        if ini.has_option("history", "lastfiles"):
            text=str(ini.get("history", "lastfiles"))
            self._lastFiles=text.strip("[']").replace("', '",",").split(",")
            if self._lastFiles==[""]:
                self._lastFiles=[]
        width=800
        height=600
        if ini.has_option("window", "width"):
            width=ini.getint("window", "width")
        if ini.has_option("window", "height"):
            height=ini.getint("window", "height")
        self.resize(QSize(width,height))
        
    def writeIni(self):
        """ write options to ini """
        if hasattr(self,"debug_calls"):
            self.calls+=["MainWindow.writeIni"]
        
        ini=self.loadIni()
        if not ini.has_section("history"):
            ini.add_section("history")
        ini.set("history", "lastfiles",self._lastFiles)
        if not ini.has_section("window"):
            ini.add_section("window")
        ini.set("window", "width",self.width())
        ini.set("window", "height",self.height())
        self.saveIni(ini)
