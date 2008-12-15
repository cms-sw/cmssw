import os.path

from GraphView import *
from GraphViewWidgets import *

from qt import *

class ModuleView(GraphView):
    """ QScrollView that holds the BoxFrames """
    def __init__(self, parent=None, name=None, fl=0,maxObjects=1000):
        """ constructor """
        self.maxObjects=maxObjects
        self._box_properties=parent._box_properties

        GraphView.__init__(self,parent,name,fl)

        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ModuleView.__init__"]
        
        self.writeIni()

        self.connect(self.parenttab,PYSIGNAL("objectSelected()"),self.objectSelected)
        self.connect(self.parenttab,PYSIGNAL("objectsSelected()"),self.showPath)

    def fillMenu(self):
        """ fill entries in MainMenu """
        GraphView.fillMenu(self)
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ModuleView.fillMenu"]
        
        self.connectionsOnOffAction = QAction("Connections &On/Off","",self.parentwindow,"connectionsOnOffAction")
        self.connectionsOnOffAction.setToggleAction(1)
        self.connectionsOnOffAction.setOn(1)
        self.connect(self.connectionsOnOffAction,SIGNAL("activated()"),self.viewChanged)

        self.treeStructureAction = QAction("&Tree structure","Ctrl+T",self.parentwindow,"treeStructureAction")
        self.treeStructureAction.setToggleAction(1)
        self.treeStructureAction.setOn(0)
        self.connect(self.treeStructureAction,SIGNAL("activated()"),self.treeStructure)
        
        self.connectionStructureAction = QAction("&Connection structure","Ctrl+C",self.parentwindow,"connectionStructureAction")
        self.connectionStructureAction.setToggleAction(1)
        self.connectionStructureAction.setOn(0)
        self.connect(self.connectionStructureAction,SIGNAL("activated()"),self.connectionStructure)
        
        self.hybridStructureAction = QAction("&Hybrid structure","Ctrl+H",self.parentwindow,"hybridStructureAction")
        self.hybridStructureAction.setToggleAction(1)
        self.hybridStructureAction.setOn(1)
        self.connect(self.hybridStructureAction,SIGNAL("activated()"),self.hybridStructure)
        
        self.treeLayoutAction = QAction("T&ree layout","Ctrl+R",self.parentwindow,"treeLayoutAction")
        self.treeLayoutAction.setToggleAction(1)
        self.treeLayoutAction.setOn(0)
        self.connect(self.treeLayoutAction,SIGNAL("activated()"),self.treeLayout)

        self.boxInBoxLayoutAction = QAction("&Box in box layout","Ctrl+B",self.parentwindow,"boxInBoxLayoutAction")
        self.boxInBoxLayoutAction.setToggleAction(1)
        self.boxInBoxLayoutAction.setOn(1)
        self.connect(self.boxInBoxLayoutAction,SIGNAL("activated()"),self.boxInBoxLayout)

        self.GraphView.insertSeparator()
        self.connectionsOnOffAction.addTo(self.GraphView)
        self.GraphView.insertSeparator()
        self.treeStructureAction.addTo(self.GraphView)
        self.connectionStructureAction.addTo(self.GraphView)
        self.hybridStructureAction.addTo(self.GraphView)
        self.GraphView.insertSeparator()
        self.treeLayoutAction.addTo(self.GraphView)
        self.boxInBoxLayoutAction.addTo(self.GraphView)

    def showMenu(self):
        """ show entries in MainMenu """
        GraphView.showMenu(self)
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ModuleView.showMenu"]
        
        self.connectionsOnOffAction.setVisible(True)
        self.treeStructureAction.setVisible(True)
        self.connectionStructureAction.setVisible(True)
        self.hybridStructureAction.setVisible(True)
        self.treeLayoutAction.setVisible(True)
        self.boxInBoxLayoutAction.setVisible(True)
        
    def hideMenu(self):
        """ hide entries in MainMenu """
        GraphView.hideMenu(self)
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ModuleView.hideMenu"]
        
        self.connectionsOnOffAction.setVisible(False)
        self.treeStructureAction.setVisible(False)
        self.connectionStructureAction.setVisible(False)
        self.hybridStructureAction.setVisible(False)
        self.treeLayoutAction.setVisible(False)
        self.boxInBoxLayoutAction.setVisible(False)
        
    def getBoxPosition(self,object,objects):
        """ calculate the position of the object in the ModuleView """
        frame_top=self._boxSpacing
        frame_left=self._boxSpacing
        mothers=[]
        for mother in object.getMothers():
            if hasattr(mother,"graphicObject") and mother.graphicObject!=[]:
                if mother.graphicObject[0] in objects:
                    mothers+=[mother.graphicObject[0]]
        if self.connectionStructureAction.isOn() and self.connectionsOnOffAction.isOn():
            mothers=[]
        if mothers!=[]:
            if self.treeLayoutAction.isOn():
                frame_left=mothers[0].x()+self._childrenIndent
                frame_top=mothers[0].y()+mothers[0].own_height+self._boxSpacing
            else:
                frame_left=mothers[0].x()+self._boxSpacing
                frame_top=mothers[0].y()+mothers[0].own_height+self._boxSpacing
        if self.connectionStructureAction.isOn() or self.hybridStructureAction.isOn():
            this_uses=[]
            this_uses_other=False
            configobjects=[]
            for o in objects:
                configobjects+=[o.obj]
            if self.connectionsOnOffAction.isOn():
                for plug in object.sinks:
                    if plug.connection.source in configobjects:
                        this_uses+=[plug.connection.source]
                        for plug2 in plug.connection.source.sources:
                            if plug2.connection.sink in configobjects:
                                this_uses_other=True
            if len(this_uses)==0 or this_uses_other:
                for entry in objects:
                    if not entry in mothers and frame_top<entry.y()+entry.height()+self._boxSpacing:
                        frame_top=entry.y()+entry.height()+self._boxSpacing
            elif len(mothers)>0:
                pass
            elif len(this_uses)>0:
                for entry in objects:
                    if frame_top<entry.y() and entry.obj in this_uses:
                        frame_top=entry.y()
            else:
                for entry in objects:
                    if frame_top<entry.y():
                        frame_top=entry.y()
            for entry in objects:
                if frame_left<entry.x()+entry.width()+self._boxSpacing and frame_top==entry.y():
                    frame_left=entry.x()+entry.width()+self._boxSpacing
                if frame_left<entry.x()+entry.width()+self._boxSpacing and entry.obj in this_uses:
                    frame_left=entry.x()+entry.width()+self._boxSpacing
        else:
            for entry in objects:
                if not entry in mothers and frame_top<entry.y()+entry.height()+self._boxSpacing:
                    frame_top=entry.y()+entry.height()+self._boxSpacing
        if frame_top>60000:
            print "ERROR: Unable to display boxes. More than 60000 pixels used."
            return None,None
        return frame_left,frame_top

    def sortConnectionStructure(self,objects):
        """ Sort objects to display connection structure """
        sorted_objects=[]
        if objects!=[]:
            object=objects[0]
        next=[]
        while objects!=[]:
            pos=0
            for plug in object.sources:
                if plug.connection.sink in objects:
                    has_all_before=True
                    for plug2 in plug.connection.sink.sinks:
                        if not plug2.connection.source==object and plug2.connection.source in objects:
                            has_all_before=False
                    if has_all_before:
                        next.insert(pos,plug.connection.sink)
                        pos+=1
            if pos>0:
                next.insert(pos,object)
            if object in objects:
                objects.remove(object)
                if not "sequence" in object.options:
                    sorted_objects+=[object]
            if next!=[]:
                object=next.pop(0)
            elif objects!=[]:
                object=objects[0]
        return sorted_objects
        
    def showEntries(self,object_list,objects,connections,box_only=False,x=0,y=0):
        """ Show ConfigObject in ModuleView """
        if self.connectionStructureAction.isOn() and self.connectionsOnOffAction.isOn():
            object_list=self.sortConnectionStructure(object_list)
        this_frame=None
        for object in object_list:
            if box_only:
                (frame_left,frame_top)=(x,y)
            else:
                (frame_left,frame_top)=self.getBoxPosition(object,objects)
            if frame_left!=None and frame_top!=None:
                this_frame=drawBox(self,object,objects,frame_left,frame_top,(self.boxInBoxLayoutAction.isOn() and not box_only),self.connectionsOnOffAction.isOn())
                if "sequence" in object.options:
                    this_frame.frame_color="darkgrey"
                if self.connectionsOnOffAction.isOn() and not box_only:
                    drawConnections(self,this_frame,connections,0)
        return this_frame

    def showPath(self):
        """ Show list of Objects in GraphView """
        self.clearObjects()
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.showPath"]
        objects=self.parenttab.selected_objects
        self.parentwindow.StatusBar.message("showing box view: "+str(len(objects))+" objects...")
        showObjects=[]
        numObjects=0
        for o in objects:
            numObjects+=1
            showObjects+=[o]
            if numObjects>self.maxObjects:
                print "ERROR: Unable to show more objects in GraphView. (maximum is set to "+str(self.maxObjects)+")"
                break
        self.showEntries(showObjects,self._objects,self._connections)
        self.updateSize()
        self.parentwindow.StatusBar.message("showing box view: "+str(len(objects))+" objects...done")

    def objectSelected(self,byClick=False):
        """ select object in GraphView """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.objectSelected"]
        
        object=self.parentwindow.selected_object
        
        entry=None
        while entry==None and not object==None:
            for o in self.allObjects():
                if o.obj==object:
                    entry=o
            if entry==None:
                object=object.getFirstMother()

# make selected object grey
        if self._selected_object!=entry:
            for o in self.allObjects():
                if o==entry:
                    o.setPaletteBackgroundColor(QColor("lightgrey"))
                    self.ensureVisible(o.x()+o.width()+self.contentsX(),o.y()+o.height()+self.contentsY(),self._boxSpacing,self._boxSpacing)
                    self.ensureVisible(o.x()+self.contentsX(),o.y()+self.contentsY(),self._boxSpacing,self._boxSpacing)
                else:
                    o.setPaletteBackgroundColor(QColor("white"))
        self._selected_object=entry

    def treeStructure(self):
        """ Show tree structure """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ModuleView.treeStructure"]

        self.treeStructureAction.setOn(True)
        self.connectionStructureAction.setOn(False)
        self.hybridStructureAction.setOn(False)
        self.viewChanged()

    def connectionStructure(self):
        """ Show connection structure """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ModuleView.connectionStructure"]

        self.treeStructureAction.setOn(False)
        self.connectionStructureAction.setOn(True)
        self.hybridStructureAction.setOn(False)
        self.viewChanged()

    def hybridStructure(self):
        """ Show hybrid structure """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ModuleView.hybridStructure"]

        self.treeStructureAction.setOn(False)
        self.connectionStructureAction.setOn(False)
        self.hybridStructureAction.setOn(True)
        self.viewChanged()

    def treeLayout(self):
        """ Show tree layout """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ModuleView.treeLayout"]

        self.treeLayoutAction.setOn(True)
        self.boxInBoxLayoutAction.setOn(False)
        self.viewChanged()

    def boxInBoxLayout(self):
        """ Show boxInBox layout """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ModuleView.boxInBoxLayout"]

        self.treeLayoutAction.setOn(False)
        self.boxInBoxLayoutAction.setOn(True)
        self.viewChanged()

    def readIni(self):
        """ read options from ini """
        GraphView.readIni(self)
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ModuleView.readIni"]
        
        ini=self.parentwindow.loadIni()
        if ini.has_option("ModuleView", "connections"):
            self.connectionsOnOffAction.setOn(ini.getboolean("ModuleView", "connections"))
        if ini.has_option("ModuleView", "structure"):
            self.treeStructureAction.setOn(str(ini.get("ModuleView", "structure"))=="tree")
            self.connectionStructureAction.setOn(str(ini.get("ModuleView", "structure"))=="connection")
            self.hybridStructureAction.setOn(not self.treeStructureAction.isOn() and not self.connectionStructureAction.isOn())
        if ini.has_option("ModuleView", "layout"):
            self.treeLayoutAction.setOn(str(ini.get("ModuleView", "layout"))=="tree")
            self.boxInBoxLayoutAction.setOn(not self.treeLayoutAction.isOn())
        if ini.has_option("ModuleView", "maxobjects"):
            self.maxObjects=ini.getint("ModuleView", "maxobjects")

    def writeIni(self):
        """ write options to ini """
        GraphView.writeIni(self)
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["ModuleView.writeIni"]
        
        ini=self.parentwindow.loadIni()
        if not ini.has_section("ModuleView"):
            ini.add_section("ModuleView")
        ini.set("ModuleView", "connections",self.connectionsOnOffAction.isOn())
        if self.treeStructureAction.isOn():
            ini.set("ModuleView", "structure","tree")
        elif self.connectionStructureAction.isOn():
            ini.set("ModuleView", "structure","connection")
        else:
            ini.set("ModuleView", "structure","hybrid")
        if self.treeLayoutAction.isOn():
            ini.set("ModuleView", "layout","tree")
        else:
            ini.set("ModuleView", "layout","boxinbox")
        ini.set("ModuleView","maxobjects",self.maxObjects)
        self.parentwindow.saveIni(ini)
