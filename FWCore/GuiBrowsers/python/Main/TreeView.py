from Icons import *

from qt import *

class ListViewItem(QListViewItem):
    """ QListViewItem with Object """
    def __init__(self,object,parent,name1,name2=None):
        """ Constructor creates QListViewItems"""
        self.obj=object
        if name2==None:
            QListViewItem.__init__(self,parent,name1)
        else:
            QListViewItem.__init__(self,parent,name1,name2)
#    def __getattr__(self,name):
#        return getattr(self.obj,name)

class TreeView(QListView):
    """ QListView """
    def __init__(self, parent=None,name=None,fl=0):
        """ constructor """
        self.parentwindow=parent.parent().parent().parent().parent()

        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["TreeView.__init__"]
        
        QListView.__init__(self,parent,name,fl)
        self.parenttab=parent
        self.setSorting(-1)
        self.addColumn("Tree View")
        self.setRootIsDecorated(True)
        self.fillMenu()
        self.readIni()
        self.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Expanding,0)

        self._objects=[]
        self._update=True

        self.connect(self.parenttab,PYSIGNAL("clearObjects()"),self.clearObjects)
        self.connect(self.parenttab,PYSIGNAL("refreshObjects()"),self.showPath)
        self.connect(self.parenttab,PYSIGNAL("objectsSelected()"),self.objectsSelected)
        self.connect(self.parenttab,PYSIGNAL("tabVisible()"),self.showMenu)
        self.connect(self.parenttab,PYSIGNAL("tabInvisible()"),self.hideMenu)
        self.connect(self,SIGNAL("selectionChanged(QListViewItem*)"),self.selectionChanged)

    def fillMenu(self):
        """ fill entries in menu """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["TreeView.fillMenu"]
        
        self.TreeView = QPopupMenu(self.parentwindow)
        self.TreeView.id=self.parentwindow.MenuBar.count()+1
        self.parentwindow.MenuBar.insertItem(QString(""),self.TreeView,self.TreeView.id)
        self.parentwindow.MenuBar.findItem(self.TreeView.id).setText("&TreeView")

        self.expandAllAction = QAction("&Expand all","Ctrl+E",self.parentwindow,"expandAllAction")
        self.connect(self.expandAllAction,SIGNAL("activated()"),self.expandAll)

        self.collapseAllAction = QAction("&Collapse all","Ctrl+L",self.parentwindow,"collapseAllAction")
        self.connect(self.collapseAllAction,SIGNAL("activated()"),self.collapseAll)
        
        self.expandAllAction.addTo(self.TreeView)
        self.collapseAllAction.addTo(self.TreeView)

    def showMenu(self):
        """ show entries in MainMenu """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["TreeView.showMenu"]
        
        self.parentwindow.MenuBar.setItemVisible(self.TreeView.id,True)
        self.expandAllAction.setVisible(True)
        self.collapseAllAction.setVisible(True)

    def hideMenu(self):
        """ hide entries in MainMenu """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["TreeView.hideMenu"]
        
        self.parentwindow.MenuBar.setItemVisible(self.TreeView.id,False)
        self.expandAllAction.setVisible(False)
        self.collapseAllAction.setVisible(False)
        
    def showListEntryRecursive(self,mother,previous,object,objects):
        """ Add Object to ListView """
        if previous==None:
            entry = ListViewItem( object, mother, object.label );
        else:
            entry = ListViewItem( object, mother, previous, object.label );
        self._objects+=[entry]
        prev=None
        for daughter in object.daughters:
            if daughter in objects:
                prev=self.showListEntryRecursive(entry,prev,objects.pop(objects.index(daughter)),objects)
        return entry

    def clearObjects(self):
        """ Clear objects """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["TreeView.clearObjects"]
            
        for entry in self._objects:
            if hasattr(entry,"obj"):
                del entry.obj
        self._objects=[]
        self.clear()
      
    def showPath(self,update=True):
        """ Show list of Objects in QListView """
        self.clearObjects()

        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["TreeView.showPath"]
        
        objects=self.parenttab.objects
        self.parentwindow.StatusBar.message("showing tree view: "+str(len(objects))+" objects...")
        objectscopy=objects[:]
        prev=None
        while objectscopy!=[]:
            prev=self.showListEntryRecursive(self,prev,objectscopy.pop(0),objectscopy)
        if update:
            so=None
            if self._objects!=[]:
                so=self._objects[0]
            self.parentwindow.emit(PYSIGNAL("selectObjects()"),(so,))
        self.parentwindow.StatusBar.message("showing tree view: "+str(len(objects))+" objects...done")

    def expandAll(self):
        """ Expand all trees """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["TreeView.expandAll"]
            
        for o in self._objects:
            o.setOpen(True)

    def collapseAll(self):
        """ Collapse all trees """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["TreeView.collapseAll"]
            
        for o in self._objects:
            o.setOpen(False)

    def selectEntry(self,entry):
        """ select entry in TreeView """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["TreeView.selectEntry"]

        if entry in self._objects:
            self._update=False
            self.setSelected(entry,True)
            self.ensureItemVisible(entry)
            self._update=True

    def objectsSelected(self):
        """ select object in TreeView """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["TreeView.objectsSelected"]
            
        object=self.parenttab.selected_object
        
        selection=None
        while object!=None and selection==None:
            for entry in self._objects:
                if entry.obj==object:
                    selection=entry
            object=object.getFirstMother()
        self.selectEntry(selection)

    def selectionChanged(self,obj):
        """ update other view when object is selected """
        if self._update:
            self.parentwindow.emit(PYSIGNAL("selectObjects()"),(obj,))

    def resizeEvent(self,e):
        """ save size to ini on resize view """
        if hasattr(self,"debug_calls"):
            self.calls+=["TreeView.resizeEvent"]
            
        QListView.resizeEvent(self,e)
        self.writeIni()

    def readIni(self):
        """ read options from ini """
        if hasattr(self,"debug_calls"):
            self.calls+=["TreeView.readIni"]
        
        ini=self.parentwindow.loadIni()
        width=200
        if ini.has_option("treeview", "width"):
            width=ini.getint("treeview", "width")
        self.resize(width,0)
        
    def writeIni(self):
        """ write options to ini """
        if hasattr(self,"debug_calls"):
            self.calls+=["TreeView.writeIni"]
        
        ini=self.parentwindow.loadIni()
        if not ini.has_section("treeview"):
            ini.add_section("treeview")
        ini.set("treeview", "width",self.width())
        self.parentwindow.saveIni(ini)
