import logging

from PyQt4.QtCore import SIGNAL
from PyQt4.QtCore import QCoreApplication
from PyQt4.QtGui import QPalette

from ConnectableWidget import *
from WidgetContainer import *
from VispaWidget import *
from BasicDataAccessor import *
from RelativeDataAccessor import *
from ParticleDataAccessor import *
from PortConnection import *
from WidgetView import *
from Exceptions import *

class BoxDecayTree(WidgetView):
    """Visualizes a decay tree using boxes to represent containers as well as their contents.
    
    Mother/daughter relations are represented by connection lines. The BoxDecayTree is automatically filled using a DataAccessor.
    """
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        WidgetView.__init__(self, parent)

        self._operationId = 0
        self._boxContentScript = ""
        self._sortBeforeArranging = True
        self._subView = None
        self._subViews = []

        self.setPalette(QPalette(Qt.black, Qt.white))

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
    
    def clear(self):
        """ Deletes all boxes in the BoxDecayTree
        """
        logging.debug(__name__ + ": clear")
        # Abort currently ongoing drawing operations
        self._operationId += 1
        WidgetView.clear(self)
        self._subViews = []

    def updateContent(self):
        """ Clear the BoxDecayTree and refill it.
        """
        logging.debug(__name__ + ": updateContent")
        self._updatingFlag = True
        self.clear()
        operationId = self._operationId
        if self._dataAccessor:
            objects = self._filter(self._dataObjects)
            if self._sortBeforeArranging:
                objects = self._sortByRelations(operationId, objects)
            self.createBoxesRecursive(operationId, objects, self)
        self._updatingFlag = False
        return operationId == self._operationId

    def createBox(self, widgetParent, container, title, text):
        """ Create a WidgetContainer or ConnectableWidget and set its properties.
        """
        if container:
            widget = WidgetContainer(widgetParent)
        else:
            widget = ConnectableWidget(widgetParent)
            widget.TEXTFIELD_FLAGS = 0
            widget.setText(text)
            widget.setShowPortNames(True)
        widget.setDragable(False)
        widget.setDeletable(False)
        widget.setAutoresizeEnabled(True, False)
#        widget.ROUNDRECT_RADIUS=0
#        widget.setColors(Qt.black,Qt.white,Qt.white)
        widget.setTitle(title)
        widget.move(widget.getDistance('leftMargin'), widget.getDistance('topMargin'))
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
        if color:
            connection.FILL_COLOR2 = color
        connection.move(2 * w1.getDistance('leftMargin'), 2 * w1.getDistance('topMargin'))
        return connection

    def createConnections(self, operationId, widgetParent):
        """ Create connection lines between objects.
        
        In BoxDecayTree default mother-daughter relations are vizualized by the connections.
        """
        if widgetParent:
            children = [w for w in widgetParent.children()
                        if isinstance(w, ConnectableWidget)]
        else:
            children = []
        for w1 in children:
            w1.setShowPortNames(False)
            for w2 in children:
                if operationId != self._operationId:
                    break
                if w1.object in self._dataAccessor.motherRelations(w2.object):
                    connectionWidget=self.createConnection(w1, 'daughterRelations', w2, 'motherRelations', None, False)
                    connectionWidget.stackUnder(w2)
                    connectionWidget.show()
        
    def arrangeBoxPosition(self, widget):
        """ Arrange box position according to mother relations.
        """
        leftMargin = widget.getDistance('leftMargin')
        topMargin = widget.getDistance('topMargin')
        x = leftMargin
        y = topMargin
        motherRelations = self._dataAccessor.allMotherRelations(widget.object)
        if widget.parent():
            vispaWidgets = [w for w in widget.parent().children() if isinstance(w, VispaWidget)]
            children = vispaWidgets[:vispaWidgets.index(widget)]
        else:
            children = []
        for w in children:
            if w.object in motherRelations:
                # place daughter box on the right of the mother box
                if x < w.x() + w.width():
                    x = w.x() + w.width() + leftMargin
                if y < w.y():
                    y = w.y()
        # resolve remaining overlaps by moving the box downwards
        for w in children:
            if not w.object in motherRelations and\
                y < w.y() + w.height():
                y = w.y() + w.height() + topMargin
        # resolve overlaps with SubViews
        if self._subView != None and widget.parent():
            subViews = [w for w in widget.parent().children() if isinstance(w, self._subView)]
        else:
            subViews = []
        for w in subViews:
            if y < w.y() + w.height():
                y = w.y() + w.height() + topMargin
        widget.move(x, y)

    def updateBoxPositions(self, widget):
        """ Arrange box positions of all objects below a certain widget.
        
        This makes sure that all boxes are in the right position after a widget is resized.
        """
        if widget.parent():
            vispaWidgets = [w for w in widget.parent().children() if isinstance(w, VispaWidget)]
            children = vispaWidgets[vispaWidgets.index(widget) + 1:]
        else:
            children = []
        for w in children:
            self.arrangeBoxPosition(w)

    def createBoxesRecursive(self, operationId, objects, widgetParent, positionName="0"):
        """ Creates a box from an object.
        
        All children of this object are created recursively.
        """
        # Process application event loop in order to accept user input during time consuming drawing operation
        QCoreApplication.instance().processEvents()
        # Abort drawing if operationId out of date
        if operationId != self._operationId:
            return None

        decayTreeChildren = []
        otherChildren = []
        for daughter in objects:
            if len(self._dataAccessor.children(daughter)) == 0 and isinstance(self._dataAccessor, ParticleDataAccessor) and self._dataAccessor.id(daughter) != None:
                decayTreeChildren += [daughter]
            else:
                otherChildren += [daughter]
        if self._subView != None and len(decayTreeChildren) > 0:
            subView = self._subView(widgetParent)
            subView.setDataAccessor(self._dataAccessor)
            subView.setDataObjects(decayTreeChildren)
            subView.updateContent()
            if isinstance(widgetParent, WidgetContainer):
                subView.move(widgetParent.getDistance('leftMargin'), widgetParent.getDistance('topMargin'))
            self._subViews += [subView]
            self.connect(subView, SIGNAL("selected"), self.onSubViewSelected)
        else:
            otherChildren += decayTreeChildren
        if self._sortBeforeArranging:
            otherChildren = self._sortByRelations(operationId, otherChildren)

        i = 0
        for object in otherChildren:
            # create box
            text = ""
            if self._boxContentScript != "":
                dataAccessorObject = BasicDataAccessorInterface(object, self._dataAccessor)
                try:
                    text = dataAccessorObject.runScript(self._boxContentScript).replace("None", "")
                except Exception, e:
                    logging.info("Error in script: " + exception_traceback())
                    text = "Error in script: " + str(e)
            widget = self.createBox(widgetParent, len(self._dataAccessor.children(object)) > 0, self._dataAccessor.label(object), text)
            child_positionName = positionName + "-" + str(i)
            self.addWidget(widget, object, child_positionName)
            i += 1
            
        if operationId != self._operationId:
            return None
        # create Connections
        self.createConnections(operationId, widgetParent)

        for widget in widgetParent.children():
            # create children objects
            if isinstance(widget, WidgetContainer):
                self.createBoxesRecursive(operationId, self._filter(self._dataAccessor.children(widget.object)), widget, positionName)

            if operationId != self._operationId:
                return None
            # resize box
            if isinstance(widget, WidgetContainer):
                widget.autosize()
                self.connect(widget, SIGNAL("sizeChanged"), self.updateBoxPositions)

            if operationId != self._operationId:
                return None
            # calculate box position
            if isinstance(widget, VispaWidget):
                self.arrangeBoxPosition(widget)

            if operationId != self._operationId:
                return None
            # draw box
            if isinstance(widget, VispaWidget):
                widget.show()
        if operationId != self._operationId:
            return None
        self.autosizeScrollArea() 

    def _sortByRelations(self, operationId, objects):
        """ Sort a list of objects by their mother/daughter relations.
        
        All daughter objects are put directly behind the mother object in the list.
        This sorting algorithm is run before the display of objects with relations.
        """
        logging.debug(__name__ + ": _sortByRelations")
        if len(objects) == 0:
            return ()
        unsortedObjects = list(objects)
        sortedObjects = []
        for object in reversed(list(objects)):
            if len(self._dataAccessor.motherRelations(object))==0:
                unsortedObjects.remove(object)
                sortedObjects.insert(0,object)
                i=0
                for child in self._dataAccessor.allDaughterRelations(object):
                    i+=1
                    if child in unsortedObjects:
                        unsortedObjects.remove(child)
                        sortedObjects.insert(i,child)
        sortedObjects+=unsortedObjects
        return tuple(sortedObjects)
        
    def closeEvent(self, event):
        self.clear()
        WidgetView.closeEvent(self, event)

    def onSubViewSelected(self, object):
        """ When item is selected in SubView forward signal.
        """
        logging.debug(self.__class__.__name__ + ": onDecayTreeItemSelected")
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
