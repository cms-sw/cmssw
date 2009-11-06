import logging

from PyQt4.QtCore import QCoreApplication,SIGNAL,Qt
from PyQt4.QtGui import QPalette

from Vispa.Main.Application import Application
from Vispa.Gui.ConnectableWidget import ConnectableWidget
from Vispa.Gui.WidgetContainer import WidgetContainer
from Vispa.Gui.VispaWidget import VispaWidget
from Vispa.Share.BasicDataAccessor import BasicDataAccessor
from Vispa.Share.BasicDataAccessor import BasicDataAccessorInterface
from Vispa.Share.RelativeDataAccessor import RelativeDataAccessor
from Vispa.Share.ParticleDataAccessor import ParticleDataAccessor
from Vispa.Gui.PortConnection import LinearPortConnection
from Vispa.Views.WidgetView import WidgetView
from Vispa.Main.Exceptions import *
from Vispa.Share.ThreadChain import ThreadChain

class BoxDecayView(WidgetView):
    """Visualizes a decay tree using boxes to represent containers as well as their contents.
    
    Mother/daughter relations are represented by connection lines. The BoxDecayView is automatically filled using a DataAccessor.
    """
    
    LABEL = "&Box Decay View"
    
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        WidgetView.__init__(self, parent)

        self._operationId = 0
        self._boxContentScript = ""
        self._sortBeforeArranging = True
        self._subView = None
        self._arrangeUsingRelationsFlag=True
        self._subViews = []
        self._leftMargin = ConnectableWidget.LEFT_MARGIN
        self._topMargin = ConnectableWidget.TOP_MARGIN

        self.setPalette(QPalette(Qt.black, Qt.white))

    def setArrangeUsingRelations(self, set):
        self._arrangeUsingRelationsFlag=set
    
    def arrangeUsingRelations(self):
        return self._arrangeUsingRelationsFlag

    def setSubView(self, view):
        self._subView = view
    
    def setSortBeforeArranging(self, set):
        self._sortBeforeArranging = set
    
    def sortBeforeArranging(self):
        self._sortBeforeArranging

    def setBoxContentScript(self, script):
        self._boxContentScript = script

    def boxContentScript(self):
        return self._boxContentScript

    def setDataAccessor(self, accessor):
        """ Sets the DataAccessor from which the boxes are created.
        
        You need to call updateContent() in order to make the changes visible.   
        """
        if not isinstance(accessor, BasicDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type BasicDataAccessor.")
        if not isinstance(accessor, RelativeDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type RelativeDataAccessor.")
        WidgetView.setDataAccessor(self, accessor)
    
    def cancel(self):
        """ Stop all running operations.
        """
        self._operationId += 1
        
    def clear(self):
        """ Deletes all boxes in the BoxDecayView
        """
        #logging.debug(__name__ + ": clear")
        WidgetView.clear(self)
        self._subViews = []

    def updateContent(self):
        """ Clear the BoxDecayView and refill it.
        """
        logging.debug(__name__ + ": updateContent")
        self.cancel()
        if self.dataAccessor() == None:
            return False
        self._updatingFlag+=1
        self.clear()
        if self.dataObject()==None:
            self._updatingFlag-=1
            return True
        operationId = self._operationId
        objects = self.applyFilter(self.dataObjects())
        if self._sortBeforeArranging and self.arrangeUsingRelations():
            thread = ThreadChain(self._sortByRelations, objects)
            while thread.isRunning():
                if not Application.NO_PROCESS_EVENTS:
                    QCoreApplication.instance().processEvents()
            objects=thread.returnValue()
        if operationId != self._operationId:
            self._updatingFlag -=1
            return False
        self.createBoxesRecursive(operationId, objects, self)
        # arrange objects which are not within a container
        BoxDecayContainer.autolayoutAlgorithm(self)
        self._updatingFlag -=1
        return operationId == self._operationId

    def createBox(self, widgetParent, container, title, text):
        """ Create a WidgetContainer or ConnectableWidget and set its properties.
        """
        if container:
            widget = BoxDecayContainer(widgetParent)
        else:
            widget = ConnectableWidget(widgetParent)
            widget.noRearangeContent()
            widget.setText(text)
            widget.textField().setOutputFlags(0)
            widget.setShowPortNames(True)
        widget.setDragable(False)
        widget.setDeletable(False)
        widget.enableAutosizing(True, False)
#        widget.ROUNDRECT_RADIUS=0
#        widget.setColors(Qt.black,Qt.white,Qt.white)
        widget.setTitle(title)
        return widget

    def createSourcePort(self, w, name, visible=True):
        """ Create a source port and set its properties.
        """
        port = w.sourcePort(name)
        if not port:
            port = w.addSourcePort(name)
            port.setDragable(False)
            port.setSelectable(False)
            if not visible:
                port.HEIGHT = 0
            port.show()
        return port

    def createSinkPort(self, w, name, visible=True):
        """ Create a sink port and set its properties.
        """
        port = w.sinkPort(name)
        if not port:
            port = w.addSinkPort(name)
            port.setDragable(False)
            port.setSelectable(False)
            if not visible:
                port.HEIGHT = 0
            port.show()
        return port

    def createConnection(self, w1, name1, w2, name2, color=None, portsVisible=True):
        """ Create a connection widget between w1 and w2.
        """
        port1 = self.createSourcePort(w1, name1, portsVisible)
        port2 = self.createSinkPort(w2, name2, portsVisible)
        connection = LinearPortConnection(w1.parent(), port1, port2)
        connection.setSelectable(False)
        connection.setDeletable(False)
        # need to move connection or hide
        # otherwise childrenRect() will be too big when called for first layout
        #connection.move(1, 50)
        #connection.hide()
        
        # found real solution:
        # call updateConnections of ConnectableWidgetOwner before 
        # childrenRect is called for the first time
        if color:
            connection.FILL_COLOR2 = color
        return connection

    def createConnections(self, operationId, widgetParent):
        """ Create connection lines between objects.
        
        In BoxDecayView default mother-daughter relations are vizualized by the connections.
        """
        for w1 in widgetParent.children():
            # Process application event loop in order to accept user input during time consuming drawing operation
            if not Application.NO_PROCESS_EVENTS:
                QCoreApplication.instance().processEvents()
            # Abort drawing if operationId out of date
            if operationId != self._operationId:
                return None
            if isinstance(w1, ConnectableWidget):
                w1.setShowPortNames(False)
                for daughter in self.dataAccessor().daughterRelations(w1.object):
                    w2 = self.widgetByObject(daughter)
                    if w2:
                        connectionWidget = self.createConnection(w1, 'daughterRelations', w2, 'motherRelations', None, False)
                        connectionWidget.stackUnder(w2)
                        #connectionWidget.show()
        return True

    def createBoxesRecursive(self, operationId, objects, widgetParent, positionName="0"):
        """ Creates a box from an object.
        
        All children of this object are created recursively.
        """
        #logging.debug(__name__ + ": createBoxesRecursive")
        # Process application event loop in order to accept user input during time consuming drawing operation
        if not Application.NO_PROCESS_EVENTS:
            QCoreApplication.instance().processEvents()
        # Abort drawing if operationId out of date
        if operationId != self._operationId:
            return None

        decayTreeChildren = []
        otherChildren = []
        for daughter in objects:
            if not self.dataAccessor().isContainer(daughter) and isinstance(self.dataAccessor(), ParticleDataAccessor) and self.dataAccessor().particleId(daughter) != None:
                decayTreeChildren += [daughter]
            else:
                otherChildren += [daughter]
        if self._subView != None and len(decayTreeChildren) > 0:
            subView = self._subView(widgetParent)
            subView.setDataAccessor(self.dataAccessor())
            subView.setDataObjects(decayTreeChildren)
            subView.updateContent()
            self._subViews += [subView]
            self.connect(subView, SIGNAL("selected"), self.onSubViewSelected)
        else:
            otherChildren += decayTreeChildren
        if self._sortBeforeArranging:
            thread = ThreadChain(self._sortByRelations, otherChildren)
            while thread.isRunning():
                if not Application.NO_PROCESS_EVENTS:
                    QCoreApplication.instance().processEvents()
            if operationId != self._operationId:
                return None
            otherChildren=thread.returnValue()

        i = 0
        for object in otherChildren:
            # create box
            text = ""
            if self._boxContentScript != "":
                dataAccessorObject = BasicDataAccessorInterface(object, self.dataAccessor())
                try:
                    text = dataAccessorObject.runScript(self._boxContentScript).replace("None", "")
                except Exception, e:
                    logging.info("Error in script: " + exception_traceback())
                    text = ""
            widget = self.createBox(widgetParent, self.dataAccessor().isContainer(object), self.dataAccessor().label(object), text)
            child_positionName = positionName + "-" + str(i)
            self.addWidget(widget, object, child_positionName)
            i += 1
            
        if operationId != self._operationId:
            return None
        # create Connections
        if not self.createConnections(operationId, widgetParent):
            return None

        for widget in widgetParent.children():
            # Process application event loop in order to accept user input during time consuming drawing operation
            if not Application.NO_PROCESS_EVENTS:
                QCoreApplication.instance().processEvents()
            # Abort drawing if operationId out of date
            if operationId != self._operationId:
                return None
            # create children objects
            if isinstance(widget, WidgetContainer):
                if not self.createBoxesRecursive(operationId, self.applyFilter(self.dataAccessor().children(widget.object)), widget, positionName):
                    return None
            if isinstance(widget, VispaWidget):
                widget.noRearangeContent(False)

            if operationId != self._operationId:
                return None
        
        for widget in widgetParent.children():
            # Process application event loop in order to accept user input during time consuming drawing operation
            if not Application.NO_PROCESS_EVENTS:
                QCoreApplication.instance().processEvents()
            # Abort drawing if operationId out of date
            if operationId != self._operationId:
                return None
            # draw box
            if isinstance(widget, VispaWidget):
                widget.show()
        if operationId != self._operationId:
            return None
        self.autosizeScrollArea()
        
        return True

    def _sortByRelations(self, objects):
        """ Sort a list of objects by their mother/daughter relations.
        
        All daughter objects are put directly behind the mother object in the list.
        This sorting algorithm is run before the display of objects with relations.
        """
        #logging.debug(__name__ + ": _sortByRelations")
        if len(objects) == 0:
            return ()
        unsortedObjects = list(objects)
        sortedObjects = []
        for object in reversed(list(objects)):
            globalMother=True
            for mother in self.dataAccessor().allMotherRelations(object):
                if mother in unsortedObjects:
                    globalMother=False
                    break
            if object in unsortedObjects and globalMother:
                unsortedObjects.remove(object)
                sortedObjects.insert(0, object)
                i = 0
                for child in self.dataAccessor().allDaughterRelations(object):
                    if child in unsortedObjects:
                        i += 1
                        unsortedObjects.remove(child)
                        sortedObjects.insert(i, child)
        sortedObjects += unsortedObjects
        return tuple(sortedObjects)
        
    def closeEvent(self, event):
        self.clear()
        WidgetView.closeEvent(self, event)

    def onSubViewSelected(self, object):
        """ When item is selected in SubView forward signal.
        """
        logging.debug(self.__class__.__name__ + ": onSubViewSelected")
        self.emit(SIGNAL("selected"), object)
        for sv in self._subViews:
            if object in sv.dataObjects():
                self._selection = "subview-" + str(self._subViews.index(sv)) + "-" + str(sv.dataObjects().index(object))

    def select(self, object):
        """ Mark an object as selected. Also in subviews.
        """
        logging.debug(self.__class__.__name__ + ": select")
        WidgetView.select(self, object)
        for sv in self._subViews:
            if object in sv.dataObjects():
                sv.select(object)
    
    def selection(self):
        """ Return the selected object. Also in subviews.
        """
        selection = WidgetView.selection(self)
        for sv in self._subViews:
            if sv.selection() != None:
                selection = sv.selection()
        return selection

    def restoreSelection(self):
        """ Restore selection. Also in subviews.
        """
        logging.debug(self.__class__.__name__ + ": restoreSelection")
        WidgetView.restoreSelection(self)
        if self._selection != None and self._selection.startswith("subview"):
            split = self._selection.split("-")
            if len(self._subViews) > int(split[1]):
                sv = self._subViews[int(split[1])]
                if len(sv.dataObjects()) > int(split[2]):
                    sv.select(sv.dataObjects()[int(split[2])])
        for sv in self._subViews:
            if object in sv.dataObjects():
                sv.select(object)
                
    def contentStartX(self):
        return 10 * self.zoomFactor()
    
    def contentStartY(self):
        return 10 * self.zoomFactor()
        
class BoxDecayContainer(WidgetContainer):
    AUTOSIZE = True
    AUTOSIZE_KEEP_ASPECT_RATIO = False
    AUTOLAYOUT_CHILDREN_ENABLED = True
    
    def __init__(self, parent=None):
        WidgetContainer.__init__(self, parent)
        
    def dataAccessor(self):
        return self.parent().dataAccessor()
    
    def widgetByObject(self, mother):
        return self.parent().widgetByObject(mother)
        
    def arrangeUsingRelations(self):
        return self.parent().arrangeUsingRelations()
        
    def autosizeScrollArea(self):
        return self.parent().autosizeScrollArea()
    
    def autolayoutChildren(self):
        self.__class__.autolayoutAlgorithm(self)
        
    #@staticmethod
    def autolayoutAlgorithm(self):
        """ Arrange box position according to mother relations.
        """
        widgetParent = self.parent()
        min_x = round(self.contentStartX())
        min_y = round(self.contentStartY())
        widgetBefore=None
        leftMargin = VispaWidget.LEFT_MARGIN
        topMargin = VispaWidget.TOP_MARGIN
        for widget in self.children():
            if isinstance(widget, VispaWidget):
                x = min_x
                y = min_y
                if self.arrangeUsingRelations():
                    for mother in self.dataAccessor().motherRelations(widget.object):
                        w = self.widgetByObject(mother)
                        if w:
                            # place daughter box on the right of the mother box
                            if x < w.x() + w.width():
                                x = w.x() + w.width() + leftMargin
                            # place right next to mother if its the first daughter
                            if w==widgetBefore:
                                y = w.y()
                widget.move(x, y)
                widgetBefore=widget
                # remember the position below all other objects as min_y
                min_y = y + widget.height() + widget.getDistance("topMargin")
        self.autosizeScrollArea()
        return True
    autolayoutAlgorithm = staticmethod(autolayoutAlgorithm)
