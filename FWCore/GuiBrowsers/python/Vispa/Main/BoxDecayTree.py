import logging

from PyQt4.QtCore import SIGNAL
from PyQt4.QtCore import QCoreApplication
from PyQt4.QtGui import QPalette

from ConnectableWidget import *
from WidgetContainer import *
from BasicDataAccessor import *
from RelativeDataAccessor import *
from ParticleDataAccessor import *
from PortConnection import *
from Workspace import *
from LineDecayTree import *
from Exceptions import *

class BoxDecayTree(Workspace):
    """Visualizes a decay tree using boxes to represent containers as well as their contents.
    
    Mother/daughter relations are represented by connection lines. The BoxDecayTree is automatically filled using a DataAccessor.
    """
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        Workspace.__init__(self, parent)

        self._accessor = None
        self._dataObjects = []
        self._operationId = 0
        self._boxContentScript = ""
        self._sortBeforeArranging = True
        self._useLineDecayTree = False
                
        self.setPalette(QPalette(Qt.black, Qt.white))

    def useLineDecayTree(self):
        return self._useLineDecayTree
    
    def setUseLineDecayTree(self, use):
        self._useLineDecayTree = use

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
        self._accessor = accessor
    
    def accessor(self):
        return self._accessor
    
    def setDataObjects(self, objects):
        """ Sets the selected object from which the boxes are created
        
        You need to call updateContent() in order to make the changes visible   
        """
        self._dataObjects = objects
        
    def dataObjects(self):
        return self._dataObjects

    def clear(self):
        """ Deletes all boxes in the BoxDecayTree
        """
        logging.debug(__name__ + ": clear")
        # Abort currently ongoing drawing operations
        self._operationId += 1
        Workspace.clear(self)

    def updateContent(self):
        """ Clear the BoxDecayTree and refill it.
        """
        logging.debug(__name__ + ": updateContent")
        self.clear()
        operationId = self._operationId
        if self._accessor:
            objects = self._dataObjects
            if self._sortBeforeArranging:
                objects = self._sortByRelations(operationId, objects)
            i = 0
            for object in objects:
                if operationId != self._operationId:
                    break
                self.createBoxRecursive(operationId, object, self, str(i))
                i += 1

    def createBox(self, widgetParent, container, title, text):
        """ Create a WidgetContainer or ConnectableWidget and set its properties.
        """
        if container:
            widget = WidgetContainer(widgetParent)
        else:
            widget = ConnectableWidget(widgetParent)
            widget.TEXTFIELD_FLAGS = 0
            widget.setText(text)
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
        connection.show()
        return connection

    def createConnections(self, operationId, widget, widgetParent):
        """ Create connection lines between objects.
        
        In BoxDecayTree default mother-daughter relations are vizualized by the connections.
        """
        widget.setShowPortNames(False)
        # create port for daughters
        if self._accessor.daughterRelations(widget.object):
            self.createSourcePort(widget, "daughterRelations", False)
        # create connections to mothers
        if widgetParent:
            children = [w for w in widgetParent.children()[: - 1]
                        if isinstance(w, VispaWidget)]
        else:
            children = []
        for w in children:
            if operationId != self._operationId:
                break
            if w.object in self._accessor.motherRelations(widget.object):
                self.createConnection(w, 'daughterRelations', widget, 'motherRelations', None, False)
        widget.raise_()
        
    def arrangeBoxPosition(self, widget):
        """ Arrange box position according to mother relations.
        """
        leftMargin = widget.getDistance('leftMargin')
        topMargin = widget.getDistance('topMargin')
        x = leftMargin
        y = topMargin
        motherRelations = self._accessor.allMotherRelations(widget.object)
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
        # resolve overlaps with LineDecayTree
        if widget.parent():
            lineDecayTrees = [w for w in widget.parent().children() if isinstance(w, LineDecayTree)]
        else:
            lineDecayTrees = []
        for w in lineDecayTrees:
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
            children= []
        for w in children:
            self.arrangeBoxPosition(w)

    def createBoxRecursive(self, operationId, object=None, widgetParent=None, id="0"):
        """ Creates a box from an object.
        
        All children of this object are created recursively.
        """
        # Process application event loop in order to accept user input during time consuming drawing operation
        QCoreApplication.instance().processEvents()
        # Abort drawing if operationId out of date
        if operationId != self._operationId:
            return None
        # create box
        text = ""
        if self._boxContentScript != "":
            dataAccessorObject = BasicDataAccessorInterface(object, self._accessor)
            try:
                text = dataAccessorObject.applyScript(self._boxContentScript).replace("None", "")
            except Exception:
                logging.info("Error in script: " + exception_traceback())
                text = "Error in script: " + exception_traceback()
        widget = self.createBox(widgetParent, len(self._accessor.children(object)) > 0, self._accessor.label(object), text)
        self.addWidget(widget, object, id)

        # create Connections
        if isinstance(widget, ConnectableWidget):
            self.createConnections(operationId, widget, widgetParent)

        # create children objects
        if isinstance(widget, WidgetContainer):
            decayTreeChildren = []
            otherChildren = []
            for daughter in self._accessor.children(object):
                if len(self._accessor.children(daughter)) == 0 and isinstance(self._accessor, ParticleDataAccessor) and self._accessor.id(daughter)!=None:
                    decayTreeChildren += [daughter]
                else:
                    otherChildren += [daughter]
            if self._useLineDecayTree and len(decayTreeChildren) > 0:
                lineDecayTree = LineDecayTree(widget)
                lineDecayTree.setDataAccessor(self._accessor)
                lineDecayTree.setDataObjects(decayTreeChildren)
                lineDecayTree.updateContent()
                lineDecayTree.move(widget.getDistance('leftMargin'), widget.getDistance('topMargin'))
            else:
                otherChildren += decayTreeChildren
            if self._sortBeforeArranging:
                otherChildren = self._sortByRelations(operationId, otherChildren)
            i = 0
            for daughter in otherChildren:
                self.createBoxRecursive(operationId, daughter, widget, id + "-" + str(i))
                i += 1

        # resize box
        if operationId == self._operationId and isinstance(widget, WidgetContainer):
            widget.autosize()
            self.connect(widget, SIGNAL("sizeChanged"), self.updateBoxPositions)

        # calculate box position
        if operationId == self._operationId:
            self.arrangeBoxPosition(widget)

        # draw box
        if operationId == self._operationId:
            widget.show()
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
        currentObject = unsortedObjects[0]
        while len(unsortedObjects) > 0:
            # Process application event loop in order to accept user input during time consuming drawing operation
            QCoreApplication.instance().processEvents()
            # Abort drawing if operationId out of date
            if operationId != self._operationId:
                break
            if currentObject in unsortedObjects:
                motherFound = False
                if len(self._accessor.motherRelations(currentObject)) > 0:
                    for mother in reversed(self._accessor.motherRelations(currentObject)):
                        if mother in unsortedObjects:
                            motherFound = True
                            unsortedObjects.remove(mother)
                            unsortedObjects.insert(0, mother)
                if motherFound:
                    currentObject = unsortedObjects[0]
                else:
                    unsortedObjects.remove(currentObject)
                    sortedObjects += [currentObject]
            else:
                if len(self._accessor.daughterRelations(currentObject)) > 0:
                    for daughter in reversed(self._accessor.daughterRelations(currentObject)):
                        if daughter in unsortedObjects:
                            if len(self._accessor.motherRelations(daughter)) == 1 and len(self._accessor.daughterRelations(daughter)) == 0:
                                unsortedObjects.remove(daughter)
                                sortedObjects += [daughter]
                            else:
                                unsortedObjects.remove(daughter)
                                unsortedObjects.insert(0, daughter)
                if len(unsortedObjects) > 0:
                    currentObject = unsortedObjects[0]
        return tuple(sortedObjects)

    def closeEvent(self,event):
        self.clear()
        Workspace.closeEvent(self,event)
