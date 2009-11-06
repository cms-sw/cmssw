import sys      # for maxint
import math

from PyQt4.QtCore import Qt, QPoint, QPointF, QRect, QSize, SIGNAL, QCoreApplication, QMimeData
from PyQt4.QtGui import QWidget, QPainter, QPolygon, QColor, QPen, QPalette, QPainterPath, QFont, QFontMetrics, QApplication, QDrag, QPixmap,QSizePolicy

import logging

from Vispa.Main.Application import Application
from Vispa.Main.Exceptions import exception_traceback
from Vispa.Gui.ConnectableWidgetOwner import ConnectableWidgetOwner
from Vispa.Gui.VispaWidget import VispaWidget
from Vispa.Gui.WidgetContainer import WidgetContainer
from Vispa.Gui.Zoomable import Zoomable
from Vispa.Share.ObjectHolder import ObjectHolder
from Vispa.Share.ThreadChain import ThreadChain
from Vispa.Views.AbstractView import AbstractView
from Vispa.Views.WidgetView import WidgetView


try:
    from pxl.algorithms import *
except Exception:
    logging.info(__name__ + ": " + exception_traceback())
    
class LineDecayView(WidgetView):
    
    LABEL = "&Line Decay View"

    DECAY_OBJECT_MIME_TYPE = "text/x-decay-object"
    
    def __init__(self, parent=None):
        WidgetView.__init__(self, parent)
        self.setAutoFillBackground(True)
        self.setPalette(QPalette(Qt.white))
        self.setAcceptDrops(True)
        self._tabController = None
        self._allNodes = {}
        self._nodeVector = None
        self._operationId = 0
        self._editable=False
        
        self._crateDecayObjectsDecaysThreadChain = ThreadChain()
        self.connect(self._crateDecayObjectsDecaysThreadChain, SIGNAL("finishedThreadChain"), self.createDecayObjectsThreadChainFinished)
        
    def decayObjectMimeType(self):
        return self.DECAY_OBJECT_MIME_TYPE
    
    def setDataObjects(self, objects):
        """ Overwrite WidgetView's function.
        
        Do not clear content. This will be done by updateContent() only if it is necessary.
        """
        AbstractView.setDataObjects(self, objects)
        self.clear()    # NO!
        
    def cancel(self):
        """ Stop all running operations.
        """
        self._operationId += 1
        
    def updateContent(self):
        logging.debug(self.__class__.__name__ +": updateContent()")
        self.cancel()
        self._updatingFlag+=1
        operationId = self._operationId
        existingWidgets = []
        for object in self.applyFilter(self.dataObjects()):
            if object == None:
                # usually this should not happen, just in case dataAccessor misbehaves
                continue
            if self.dataAccessor().isContainer(object):
                # event or event view
                eventWidget = self.createLineDecayContainer(object)
                existingWidgets += [eventWidget]
                if self.dataAccessor():
                    for childObject in self.applyFilter(self.dataAccessor().children(object)):
                        if self.dataAccessor().isContainer(childObject):
                            self.createLineDecayContainer(childObject, object)
            else:
                # particle
                particleWidget = ParticleWidget(self, ParticleWidget.NONE, "", "")
                particleWidget.setMinimumSize(DecayLine.DEFAULT_LENGTH, 40)
                particleWidget.setColors(QColor('white'), QColor('white'), QColor('white'))
                particleWidget.setSelectable(True)
                particleWidget.setObject(object)
                decayLine = DecayLine(particleWidget, QPoint(5, 20), QPoint(particleWidget.width() -5, 20))
                if self.dataAccessor():
                    decayLine.setLabel(self.dataAccessor().label(object))
                    decayLine.setColor(self.dataAccessor().color(object))
                    decayLine.setLineStyle(self.dataAccessor().lineStyle(object))
                else:
                    decayLine.setLabel("Particle")
                    
                particleWidget.setDecayObject(decayLine)
                existingWidgets += [particleWidget]
                
        for child in self.children():
            if operationId != self._operationId:
                self._updatingFlag -=1
                return False
            if hasattr(child, "object") and not child in existingWidgets:
                # remove widgets of objects which no longer exist
                child.setDeletable(True)
                child.delete()
            elif hasattr(child, "createDecayObjectsFromPxlObject"):
                # every cycle of this loop takes long time
                # so process window events
                #if not Application.NO_PROCESS_EVENTS:
                #    QCoreApplication.instance().processEvents()
                #self._crateDecayObjectsDecaysThreadChain.addCommand(child.createDecayObjectsFromPxlObject)
                child.createDecayObjectsFromPxlObject(operationId)
                child.setDeletable(False)
                child.setDragable(False)
         
        self.createDecayObjectsThreadChainFinished(None)
        
        self._updatingFlag-=1
        return True
    
    def createDecayObjectsThreadChainFinished(self, result):
        #logging.debug(self.__class__.__name__ +": createDecayObjectsThreadChainFinished()")
        if not self._editable:
            self.autolayout()
            
    def autolayout(self):
        #logging.debug(self.__class__.__name__ +": autolayout()")
        margin = 10 * self.zoomFactor()
        x = margin
        y = margin
        for child in self.children():
            if isinstance(child, QWidget):
                if isinstance(child, LineDecayContainer):
                    child.autolayout()
                child.move(x, y)
                y += child.height() + margin
                
    def childFinishedAutolayouting(self):
        #logging.debug(self.__class__.__name__ +": childFinishedAutolayouting()")
        self.autosizeScrollArea()
        #for child in self.children():
        #    if isinstance(child, LineDecayContainer):
        #        child.autosize()
    
    def lineDecayContainer(self, object):
        """ Returns the widget component representing the given pxl object.
        """
        if not object:
            return None
        objectId = self.dataAccessor().id(object)
        
        for child in self.children():
            if isinstance(child, LineDecayContainer):
                if self.dataAccessor().id(child.object()) == objectId:
                    return child
                subChild = child.childContainer(objectId)
                if subChild:
                    return subChild
        return None
    
    def createLineDecayContainer(self, object, objectMother=None):
        if not object:
            return None
        
        existingWidget = self.lineDecayContainer(object)
        if existingWidget:
            # It may happen that an container exists but at the wrong hierarchical level.
            # e.g. if in TreeView after an event view the superior event is selected.
               
            if objectMother:
                parentWidget = self.lineDecayContainer(objectMother)
            else:
                parentWidget = self
            oldParent = existingWidget.parent()
                
            if parentWidget != oldParent:
                self.disconnect(existingWidget, SIGNAL("finishedAutolayouting"), oldParent.childFinishedAutolayouting)
                existingWidget.setParent(parentWidget)
                self.connect(existingWidget, SIGNAL("finishedAutolayouting"), parentWidget.childFinishedAutolayouting)
                existingWidget.setVisible(True)
                
            return existingWidget
        
        if objectMother:
            parentWidget = self.lineDecayContainer(objectMother)
            if not parentWidget:
                logging.warning(self.__class__.__name__ +": createLineDecayContainer() - Cannot add child object to given object. Object does not belong to this "+ self.__class__.__name__ +".")
                return
            
            lineDecayView = parentWidget.createChildContainer(object)
        else:
        # parentWidget == self:
            lineDecayView = LineDecayContainer(self)
            self.connect(lineDecayView, SIGNAL("finishedAutolayouting"), self.childFinishedAutolayouting)
            lineDecayView.setPxlObject(object)
        
        self.connect(lineDecayView, SIGNAL("selected"), self.onSelected)
        return lineDecayView
        
    def setTabController(self, controller):
        """ Sets tab controller.
        """
        WidgetView.setTabController(self, controller)
        self.connect(self, SIGNAL("selected"), controller.onSelected)
        
    def tabController(self):
        """ Return tab controller.
        
        See setTabController()
        """
        parent=self
        while parent!=None:
            if hasattr(parent,"controller"):
                return parent.controller()
            parent=parent.parent()
        return None

    def pxlEvent(self):
        return self._pxlEvent
    
    def onSelected(self, object):
        """ When item is selected in SubView forward signal.
        """
        #logging.debug(self.__class__.__name__ + ": onSelected()")
        self.emit(SIGNAL("selected"), object)
        
    def setEditable(self, editable):
        self._editable=editable
        for child in self.children():
            if isinstance(child, LineDecayContainer):
                child.setEditable(editable)
                
    def editable(self):
        return self._editable
        
class LineDecayContainer(WidgetContainer, ObjectHolder):
    """ Represents an Event or  EventView
    """
    # inherited properties
    AUTOSIZE = True
    AUTOSIZE_KEEP_ASPECT_RATIO = False
    AUTOLAYOUT_CHILDREN_ENABLED = True
    AUTOSIZE_ADJUST_CONTAINER_POSITION = False
    WIDTH = 300
    HEIGHT = 300
    
    # new properties
    NO_DECORATIONS_ABOVE_NUMBER_OF_OBJECTS = 10000
    
    def __init__(self, parent):
        logging.debug(self.__class__.__name__ +": __init__()")
        self._subWidgetStartY = 0
        ObjectHolder.__init__(self)
        WidgetContainer.__init__(self, parent)
        self.setEditable(parent.editable())
        self.setSelectable(True)
        
        self._selectedList = []
        self._hoveredObject = None
        self.setMouseTracking(True)     # receive mouse events even if no button is pressed
        self._titleClicked = False
        
        self._pxlObject = None
        self._existingObjectIds = []
        self._particlesDict = {}
        self._threadChain = ThreadChain()
        self.setTitle(" ")      # calculate correct height
        self.show()
        self.connect(self._threadChain, SIGNAL("finishedThreadChain"), self.autolayoutThreadFinished)
        if isinstance(parent, LineDecayContainer):
            self.connect(self, SIGNAL("finishedAutolayouting"), parent.childFinishedAutolayouting)

    def applyFilter(self,objects):
        """ Redirect filtering to parent.
        """
        return self.parent().applyFilter(objects)
            
    def dataObjects(self):
        """ Do not filter widget but rather the pxl objects
        """
        return self._dataObjects
            
    def setEditable(self, editable):
        self._editable = editable
        self.setAcceptDrops(editable)
        # make event views freely moveable
        self.enableAutolayoutChildren(not editable)
        for child in self.children():
            if isinstance(child, LineDecayContainer):
                child.setEditable(editable)
        
    def editable(self):
        return self._editable
        
    def setAcceptDrops(self, accept):
        """ Sets whether this view accepts drops and forwards the information to sub LineDecayContainers.
        """
        WidgetContainer.setAcceptDrops(self, accept)
        for child in self.children():
            if isinstance(child, LineDecayContainer):
                child.setAcceptDrops(accept)

    def setDataAccessor(self, accessor):
        """ Sets the DataAccessor from which the lines are created.
        
        You need to call updateContent() in order to make the changes visible.   
        """
        if not isinstance(accessor, BasicDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type BasicDataAccessor.")
        if not isinstance(accessor, RelativeDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type RelativeDataAccessor.")
        if not isinstance(accessor, ParticleDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type ParticleDataAccessor.")
        AbstractView.setDataAccessor(self, accessor)
        
    def sizeHint(self):
        """ If there is only one container in parent increase it so it fills whole space.
        """
        if not self._editable:
            return WidgetContainer.sizeHint(self)
        
        minWidth = 0
        minHeight = 0
        if not self.collapsed() and self.parent().dataObjectsCount() == 1:
            margin = 10
            minWidth = (self.parent().width() - 2 * margin) / self.zoomFactor()
            minHeight = (self.parent().height() - 2 * margin) / self.zoomFactor()
            
        increaseSize = 0
        if not self.collapsed():
            # make it easier to drop particles into container
            increaseSize = 30
            
        sizeHint = WidgetContainer.sizeHint(self)
        return QSize(max(sizeHint.width() + increaseSize, minWidth), max(sizeHint.height() + increaseSize, minHeight))

    def childrenRect(self):
        """ Overwrites QWidget's method and includes DecayObjects in addition to QWidgets.
        """
        minX = 0
        minY = 0
        maxX = 0
        maxY = 0
        if not self.collapsed():
            first = True
            for object in self.dataObjects():
                if not isinstance(object, DecayNode):
                    continue
                if first or minX > object.x():
                    minX = object.x()
                if first or minY > object.y():
                    minY = object.y()
                if first or maxX < object.x():
                    maxX = object.x()
                if first or maxY < object.y():
                    maxY = object.y()
                first = False
            minX *= self.zoomFactor()
            minY *= self.zoomFactor()
            maxX = (maxX * self.zoomFactor() + self.getDistance("rightMargin"))
            maxY = (maxY * self.zoomFactor() + self.getDistance("bottomMargin"))
        return QRect(minX, minY, maxX - minX, maxY - minY).united(WidgetContainer.childrenRect(self))

    def children(self):
        return WidgetContainer.children(self) + [node for node in self.dataObjects() if isinstance(node, DecayNode)]
            
    def deselectAllWidgets(self, exception=None):
        """ Overwrite VispaWidgetOwner's method so in addition deselectAllObjects() is called.
        """
        self.deselectAllObjects()
        WidgetContainer.deselectAllWidgets(self, exception)
    
    def setZoom(self, zoom):
        """ Sets zoom of this widget and of it's children.
        """
        WidgetContainer.setZoom(self, zoom)
        
        for child in self.dataObjects():
            if isinstance(child, Zoomable):
                child.setZoom(zoom)
        self.update()
        
#    def dataObjectsNodePriority(self):
#        return [obj for obj in self.dataObjects() if isinstance(obj, DecayNode)] + [obj for obj in self.dataObjects() if not isinstance(obj, DecayNode)]
        
    def setPxlObject(self, object):
        self._pxlObject = object
        if self.dataAccessor():
            label = self.dataAccessor().label(object)
            if label:
                self.setTitle(label)

    def object(self):
        return self._pxlObject
        
    def tabController(self):
        """ Return tab controller.
        
        See setTabController()
        """
        return self.parent().tabController()
    
    def dataAccessor(self):
        return self.parent().dataAccessor()
    
    def childContainer(self, objectId):
        """ Returns the widget component representing the given pxl object.
        """
        if not objectId:
            return None
        
        for child in self.children():
            if isinstance(child, LineDecayContainer):
                if self.dataAccessor().id(child.object()) == objectId:
                    return child
                subChild = child.childContainer(object)
                if subChild:
                    return subChild
        return None
    
    def createChildContainer(self, object, pos=None):
        lineDecayView = LineDecayContainer(self)
        lineDecayView.setPxlObject(object)
        
        if not pos:
            margin = 10 * self.zoomFactor()
            pos = QPoint(margin, margin)
        else:
            pos -= QPoint(0.5* lineDecayView.width(), 0.5* lineDecayView.height())
        
        lineDecayView.move(pos)
        return lineDecayView
    
    def dragEnterEvent(self, event):
        """ Accepts drag enter event if module is dragged.
        """
        #logging.debug(self.__class__.__name__ + ": dragEnterEvent()")
        if not self.tabController():
            return
        if event.mimeData().hasFormat(LineDecayView.DECAY_OBJECT_MIME_TYPE):
                event.acceptProposedAction()
        
    def dropEvent(self, event):
        """ Handle drop of module.
        """
        logging.debug(self.__class__.__name__ + ": dropEvent()")
        if not self.tabController():
            return
        
        if event.mimeData().hasFormat(LineDecayView.DECAY_OBJECT_MIME_TYPE):
            dropType = str(event.mimeData().data(LineDecayView.DECAY_OBJECT_MIME_TYPE))
            pos = event.pos() / self.zoomFactor()
            newObject = None
            if dropType == "Node":
                newObject = self.addDecayNode(pos)
            elif dropType == "EventView":
                newObject = self.createChildContainer(self.object().createEventView(), pos)
                # connect selected signal to parent
                parent=self
                while hasattr(parent,"parent"):
                    if hasattr(parent,"onSelected"):
                        self.connect(newObject, SIGNAL("selected"), parent.onSelected)
                        break
                    parent=parent.parent()
            else:
                newObject = self.addDecayObject(None,dropType, pos)
                
            if newObject:
                event.acceptProposedAction()
                self.update(newObject.boundingRect())
                if self._editable:
                    self.autosize()     # instead of full autolayout
                self.deselectAllWidgets()
                self.select(newObject)
                if self.tabController():
                    self.tabController().setModified()
                    if hasattr(self.tabController(),"updateTreeView"):
                        self.tabController().updateTreeView()
                    
    def addObject(self, object):
        """ Supposed to replace part of addDecayObject getting a pxl object as argument.
        """
        pass
        
    def addDecayObject(self, object=None, objectType=None, pos=None):
        """ This function adds a decay object to this view.
        
        object or objectType may either be a valid pxl object (e.g. a particle) or a string specifiying the object type.
        pos is in 100% coordinates.
        """
        # TODO: split this function is easy understandable parts
        #logging.debug(self.__class__.__name__ +": addDecayObject()")
        if not pos:
            pos = QPoint(10, 10)
        else:
            pos -= QPoint(0.5 * DecayLine.DEFAULT_LENGTH, 0)
        
        if not object and not objectType:
            # this is nothing
            return None
        elif object:
            # argument is a (pxl) object
            # TODO: add check objectOrObjectType is a valid pxl object
            if self.dataAccessor():
                dataObjectId = self.dataAccessor().id(object)
                # id() is slow, so ids of existing objects are stored
                if dataObjectId in self._existingObjectIds:
                    # there is already a DecayObject for this object
                    return None
        
        newObject = None
        if objectType:
            #if self.dataAccessor() and objectOrObjectType in ["Particle", "Electron", "Muon", "e", "electron_neutrino", "muon", "muon_neutrino", "tau", "tau_neutrino", "u", "d", "c", "s", "t", "b", "photon", "W+", "W-", "Z", "gluon", "Higgs", "UHECR"]:
                object = self.dataAccessor().createParticle()
                object.setName(objectType)  # use this name to find id
                particleId = self.dataAccessor().particleId(object)
                if particleId != None:
                    object.setParticleId(particleId)
                particleName = self.dataAccessor().defaultName(object)
                if particleName:
                    # normalize name using id
                    object.setName(particleName)
                object.setCharge(self.dataAccessor().charge(object))
                dataObjectId = self.dataAccessor().id(object)       # uuid
                self._pxlObject.setObject(object)
        
                
        # no elif because object might be created in if-case above
        if object:
            self._existingObjectIds.append(str(dataObjectId))
            newObject = DecayLine(self, QPoint(pos), QPoint(pos.x() + DecayLine.DEFAULT_LENGTH, pos.y()))
            newObject.setObject(object)
            if self.dataAccessor():
                newObject.setLabel(self.dataAccessor().label(object))
                newObject.setColor(self.dataAccessor().color(object))
                newObject.setLineStyle(self.dataAccessor().lineStyle(object))
            self._particlesDict[object] = newObject
            
            # map pxl relations to gui components
            # TODO: try to implement without unite(), so DecayOjbect directly uses correct nodes
            if self.dataAccessor():
                for daughter in self.dataAccessor().daughterRelations(object):
                    if daughter in self._particlesDict.keys():
                        daughterDecayObject = self._particlesDict[daughter]
                        oldDaughter = newObject.daughterNode()
                        if daughterDecayObject.motherNode().unite(oldDaughter):
                            self.removeObject(oldDaughter)
                for mother in self.dataAccessor().motherRelations(object):
                    if mother in self._particlesDict.keys():
                        motherDecayObject = self._particlesDict[mother]
                        oldMother = newObject.motherNode()
                        if motherDecayObject.daughterNode().unite(oldMother):
                            self.removeObject(oldMother)
        
        if newObject:
            self.appendObject(newObject)
        return newObject
    
    def addDecayNode(self, pos):
        newObject = DecayNode(self, pos)
        return self.appendObject(newObject)
    
    def createDecayObjectsFromPxlObject(self, operationId):
        """ Creates DecayObjects for all particles in the set pxl object.
        
        In addition this function is called on all child LineDecayContainers.
        """
        self._operationId=operationId
        #logging.debug(self.__class__.__name__ +": createDecayObjectsFromPxlObject()")
        if self._pxlObject and self.dataAccessor():
            if operationId != self._operationId:
                self._updatingFlag -=1
                return False
            for childObject in self.applyFilter(self.dataAccessor().children(self._pxlObject)):
                if not self.dataAccessor().isContainer(childObject):
                    self.addDecayObject(childObject)
                
        for child in self.children():
            #if not Application.NO_PROCESS_EVENTS:
            #    QCoreApplication.instance().processEvents()
            if operationId != self._operationId:
                self._updatingFlag -=1
                return False
            if isinstance(child, LineDecayContainer):
                child.createDecayObjectsFromPxlObject(operationId)
            
    def decayObject(self, pxlObject):
        """ Returns the DecayObject which represents the given pxlObject or None if there is no such one.
        
        This function is to slow for massive usage with many dataObjects as it loops over all dataObjects.
        """
        for decayObject in self.dataObjects():
            if decayObject.object() == pxlObject:
                return decayObject
        return None
    
    def select(self, decayObject):
        if type(decayObject) == type(True):
            WidgetContainer.select(self, decayObject)
        elif not decayObject in self._selectedList:
        #if type(decayObject) != type(True) and not decayObject in self._selectedList:
            try:
                # widgetSelected() usually expects VispaWidget as argument
                # needed to prevent multiple selects in different containers
                # in case of problems, replace decayObect by None and emit selected signal (see below)
                self.parent().widgetSelected(decayObject)
                
                if decayObject in self.dataObjects():
                    self.dataObjects().remove(decayObject)
                    self.dataObjects().insert(0, decayObject)
                    self._selectedList.append(decayObject)
                    self.update(decayObject.boundingRect())
            except ValueError:
                logging.error(self.__class__.__name__ +": select() - Tried to remove non data object from data objects list. This is not supposed to happen. Check it.")
            
            # no need to emit selected signal as long as
            # widgetSelected(decayObject) works (see above)
            #self.emit(SIGNAL("selected"), decayObject.object())
            
    def deselectAllObjects(self):
        for object in self._selectedList:
            self.update(object.boundingRect())
        self._selectedList = []
        
    def objectMoved(self, object, oldBoundingRect=None):
        rect = object.boundingRect()
        if oldBoundingRect:
            rect = rect.unite(oldBoundingRect)
        self.update(rect)
        
    def paint(self, painter):
        WidgetContainer.paint(self, painter)
        
        if self.collapsed():
            # don't paint if container is collapsed
            return
        
        generalPaintMode = 0x0
        if len(self.dataObjects()) > self.NO_DECORATIONS_ABOVE_NUMBER_OF_OBJECTS:
            generalPaintMode = DecayObject.PAINT_MODE_NO_DECORATIONS
        
        if self.dataObjectsCount() > 50:
            painter.setRenderHint(0)
            
        for object in reversed(self.dataObjects()):
            if isinstance(object, DecayLine):
                paintMode = generalPaintMode
                if object in self._selectedList:
                    paintMode |= DecayObject.PAINT_MODE_SELECTED
                if object == self._hoveredObject:
                    paintMode |= DecayObject.PAINT_MODE_HOVERED
                object.paint(painter, paintMode)
        for object in self.dataObjects():
            if isinstance(object, DecayNode):
                paintMode = generalPaintMode
                if object in self._selectedList:
                    paintMode |= DecayObject.PAINT_MODE_SELECTED
                if object == self._hoveredObject:
                    paintMode |= DecayObject.PAINT_MODE_HOVERED
                object.paint(painter, paintMode)
        
    def mousePressEvent(self, event):
        if self.isTitlePoint(event.pos()):
            WidgetContainer.mousePressEvent(self, event)
            self._titleClicked = True
            return
        self.select(False)
        self._titleClicked = False
        self.deselectAllWidgets()
        
        toSelectObject = None
        for object in self.dataObjects():
            if object.containsPoint(event.pos()):
                if isinstance(object, DecayNode):
                    # prefere nodes over other DecayObjects (especially DecayLines)
                    toSelectObject = object
                    break
                elif not toSelectObject:
                    toSelectObject = object
        if toSelectObject:
            toSelectObject.select(event.pos())
            self.select(toSelectObject)
            return      # select 1 object at most
        WidgetContainer.mousePressEvent(self, event)
        
    def mouseMoveEvent(self, event):
        if self._titleClicked and bool(event.buttons() & Qt.LeftButton) and self._editable:
            WidgetContainer.mouseMoveEvent(self, event)
            return
        if not bool(event.buttons()):
            # no button pressed -> hovering
            to_hover_object = None
            for object in self.dataObjects():
                if object.containsPoint(event.pos()):
                    if isinstance(object, DecayNode):
                        # prefere nodes over other DecayObjects (especially DecayLines)
                        to_hover_object = object
                        break
                    elif not to_hover_object:
                        to_hover_object = object
            if to_hover_object:
                if to_hover_object != self._hoveredObject:
                    previously_hovered_object = self._hoveredObject
                    self._hoveredObject = to_hover_object
                    self.update(self._hoveredObject.boundingRect())
                    if previously_hovered_object:
                        # make sure hovered mode is removed if hovered object changed
                        self.update(previously_hovered_object.boundingRect())
            elif self._hoveredObject:
                self.update(self._hoveredObject.boundingRect())
                self._hoveredObject = None
            
        elif len(self._selectedList) > 0 and event.buttons() & Qt.LeftButton and self._editable:
            # selection
            self._selectedList[0].move(event.pos())
            
    def mouseReleaseEvent(self, event):
        """ Join nodes if they belong to objects with relations.
        """
        if len(self._selectedList) > 0 and isinstance(self._selectedList[0], DecayNode):
            # unite DecayNodes
            selectedObject = self._selectedList[0]
            dataObjects = self.dataObjects()[:]
            for obj in dataObjects:
                if obj != self._selectedList[0] and isinstance(obj, DecayNode) and obj.containsPoint(event.pos()):
                    hasRelations=self.dataAccessor().hasRelations(self.object())
                    for decayLine in self._selectedList[0].dataObjects()+obj.dataObjects():
                        hasRelations=hasRelations and self.dataAccessor().hasRelations(decayLine.object())
                    if not hasRelations:
                        continue
                    # selectedObject.unite(obj) # this may lead to image errors, so do it the other way
                    obj.unite(self._selectedList[0])
                    self.removeObject(selectedObject)
                    self._selectedList.append(obj)
                    if self.tabController():
                        self.tabController().setModified()
                    
                    
    def keyPressEvent(self, event):
        """ Calls delete() method if backspace or delete key is pressed when widget has focus.
        """
        if (event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete) and self._editable:
            if self.isSelected():
                self.delete()
            elif len(self._selectedList) > 0:
                self.removeObject(self._selectedList[0])
            
            if self.tabController():
                self.tabController().setModified()
                if hasattr(self.tabController(),"updateTreeView"):
                    self.tabController().updateTreeView()
    
    def delete(self):
        if WidgetContainer.delete(self) and hasattr(self.parent(), "object"):
            self.parent().object().removeObject(self.object())
            self._pxlObject = None
    
    def removeObject(self, decayObject):
        if not decayObject in self.dataObjects():
            return
        self.update(decayObject.boundingRect())
        if decayObject.delete():
            if decayObject in self._selectedList:
                self._selectedList.remove(decayObject)
            if decayObject.object():
                if self.dataAccessor():
                    id = self.dataAccessor().id(decayObject.object())
                    if id in self._existingObjectIds:
                        self._existingObjectIds.remove(id)
                self._particlesDict.pop(decayObject.object(), None)
                self._pxlObject.removeObject(decayObject.object())
            ObjectHolder.removeObject(self, decayObject)
            #self.tabController().updateContent()        # not here, can create infinite loop

    def autolayout(self):
        logging.debug(self.__class__.__name__ +": autolayout() - %s" % str(self.title()))
        if self._threadChain.isRunning():
            logging.info(self.__class__.__name__ +": autolayout() - Autolayout thread is already running. Aborting...")
            return
        
        self._autolayoutingChildren = []
        try:
            self._nodeVector = NodeVector()
            self._allNodes = {}
            for object in self.dataObjects():
                if isinstance(object, DecayLine):
                    # make sure both nodes of particle are stored in self._nodeVector
                    # and set relations between these nodes
                    if not object.motherNode() in self._allNodes.keys():
                        # create new node
                        motherNode = Node()
                        motherNode.position = Vector2(0, 0)
                        motherNode.isVertex = False
                        self._allNodes[object.motherNode()] = motherNode
                        newMother = True
                    else:
                        # use same node again, if it was already created for another particle
                        motherNode = self._allNodes[object.motherNode()]
                        newMother = False
                    if not object.daughterNode() in self._allNodes.keys():
                        daughterNode = Node()
                        daughterNode.position = Vector2(0, 0)
                        daughterNode.isVertex = False
                        self._allNodes[object.daughterNode()] = daughterNode
                        newDaughter = True
                    else:
                        daughterNode = self._allNodes[object.daughterNode()]
                        newDaughter = False 
                
                    # the important step: set mother and daughter relations
                    # between both nodes of _one_ particle
                    motherNode.children.append(daughterNode)
                    daughterNode.mothers.append(motherNode)
                
                    # only append nodes if they were newly created
                    if newMother:
                        self._nodeVector.append(motherNode)
                    if newDaughter:
                        self._nodeVector.append(daughterNode)
            
            nodeVectorSize = self._nodeVector.size()
            if nodeVectorSize > 1:
                adhoc = nodeVectorSize > 40
                autolayouter = AutoLayout()
                #logging.debug(self.__class__.__name__ +": calling pxl::AutoLayout.init() with "+str(self._nodeVector.size())+" Particles")
                autolayouter.init(self._nodeVector)
                logging.debug(self.__class__.__name__ +": calling pxl::AutoLayout.layout(%s) with %d Particles" % (str(adhoc), nodeVectorSize))
                if adhoc:
                    autolayouter.layout(False)
                else:
                    autolayouter.layout(True)
            self.autolayoutThreadFinished(None)
        except Exception:
            logging.error(__name__ + ": Pxl Autolayout not found: " + exception_traceback())
            return
        
        for child in self.children():
            if isinstance(child, LineDecayContainer):
                self._autolayoutingChildren.append(child)
                child.autolayout()
                        
    def autolayoutThreadFinished(self, result):
        logging.debug(self.__class__.__name__ +" autolayoutThreadFinished() - %s" % str(self.title()))
        if self._threadChain.isRunning():
            logging.info(self.__class__.__name__ +": autolayoutThreadFinished() - Thread is still running. Aborting...")
            return
        minOrphanY = 0
        maxNonOrphanY = 0
        firstMinOrphanY = True
        firstMaxNonOrphanY = True
        minY = 0
        maxY = 0
        for i in range(len(self._nodeVector)):
            if i == 0 or self._nodeVector[i].position.y < minY:
                minY = self._nodeVector[i].position.y
            if self._nodeVector[i].position.y > maxY:
                maxY = self._nodeVector[i].position.y
                
            #print "len_mothers, len_children", len(self._nodeVector[i].mothers), len(self._nodeVector[i].children),
            if (len(self._nodeVector[i].mothers) == 1 and len(self._nodeVector[i].children) == 0) or (len(self._nodeVector[i].mothers) == 0 and len(self._nodeVector[i].children) == 1):
                # orphan particles:
                # nodes have exactly one relation
                # eighter node is mother or daughter (left and right side of particle)
                #print " orphan"
                if firstMinOrphanY or self._nodeVector[i].position.y > minOrphanY:
                    minOrphanY = self._nodeVector[i].position.y
                    firstMinOrphanY = False
            else:
                # non orphans
                #print "non orphan"
                if firstMaxNonOrphanY or self._nodeVector[i].position.y < maxNonOrphanY:
                    #print "   new max non orphan", self._nodeVector[i].position.y
                    maxNonOrphanY = self._nodeVector[i].position.y
                    #print "   new max non orphan", maxNonOrphanY
                    firstMaxNonOrphanY = False
                    
        #print "minOrphanY, maxNonOrphanY", minOrphanY, maxNonOrphanY
            
        xOffset = -30
        yOffset = -minY + (self.getDistance("titleFieldBottom") + 4* self.getDistance("topMargin")) / self.zoomFactor()
        #print "yOffset ", yOffset 
        for decayNode in self.dataObjects():
            if isinstance(decayNode, DecayNode) and decayNode in self._allNodes.keys():
                #pxlNode = self._allNodes[decayNode]
                #if (len(nodeVector[i].mothers) == 1 and len(nodeVector[i].children) == 0) or (len(nodeVector[i].mothers) == 0 and len(nodeVector[i].children) == 1):    # orphan
                #    yOffset += maxNonOrphanY - minOrphanY + 30
                decayNode.setPosition(QPoint(self._allNodes[decayNode].position.x + xOffset, self._allNodes[decayNode].position.y + yOffset))
            
        # arrange potential sub-views under decay tree
        self._subWidgetStartY = (maxY - minY) + 1.0* (self.getDistance("titleFieldBottom") + self.getDistance("topMargin") + self.getDistance("topMargin")) / self.zoomFactor()
        if self.dataObjectsCount() > 0:
            # add some pixels for last particle
            self._subWidgetStartY += 20
            
        if len(self._autolayoutingChildren) == 0:
            self.autolayoutPostprocess()
            
    def contentStartY(self):
        return self._subWidgetStartY * self.zoomFactor()
        
    def childFinishedAutolayouting(self):
        logging.debug(self.__class__.__name__ +": childFinishedAutolayouting() - %s" % str(self.title()))
        if self.sender() and isinstance(self.sender(), LineDecayContainer):
            child = self.sender()
            if child in self._autolayoutingChildren:
                self._autolayoutingChildren.remove(child)
        
        if len(self._autolayoutingChildren) == 0 and not self._threadChain.isRunning():
            # wait until all children are done
            self.autolayoutPostprocess()
            
    def autolayoutPostprocess(self):
        if not self.autolayoutChildrenEnabled():
            # autosize() calls autolayoutChildren() anyway
            # but: even if in edit-mode and user shall be able to freely move event views
            # on manually calling autolayout children should be autolayouted as well
            # this function positionizes children, it does not call pxl autolayouter on children
            self.autolayoutChildren()
        self.autosize()
        self.emit(SIGNAL("finishedAutolayouting"))

        
class DecayObject(Zoomable):
    CONTAINS_AREA_SIZE = 4
    PAINT_MODE_SELECTED = 0x1
    PAINT_MODE_HOVERED = 0x2
    PAINT_MODE_NO_DECORATIONS = 0x4
    
    def __init__(self, parent=None):
        Zoomable.__init__(self)
        self._parent = parent
        if isinstance(self._parent, Zoomable):
            self.setZoom(self._parent.zoom())
    
    def parent(self):
        return self._parent
    
    def paint(self, painter, paintMode=0x0):
        raise NotImplementedError
    
    def boundingRect(self):
        raise NotImplementedError
    
    def containsPoint(self, pos):
        raise NotImplementedError
    
    def select(self, pos=None, selected=True):
        raise NotImplementedError
    
    def containsAreaSquareRect(self, position):
        return QRect( (position - QPoint(self.CONTAINS_AREA_SIZE, self.CONTAINS_AREA_SIZE)*0.5 ) * self.zoomFactor(), QSize(self.CONTAINS_AREA_SIZE, self.CONTAINS_AREA_SIZE) * self.zoomFactor() + QSize(1, 1))
    
    def move(self, pos):
        raise NotImplementedError
    
    def delete(self):
        pass
    
    def object(self):
        return None
    
    
class DecayNode(DecayObject, ObjectHolder):
    CONTAINS_AREA_SIZE = 8
    TYPE_MOTHER = 0
    TYPE_DAUGHTER = 1
    
    def __init__(self, parent, position):
        DecayObject.__init__(self, parent)
        ObjectHolder.__init__(self)
        self.setExclusiveMode(True)
        self._position = QPoint(position)   # copy
        self._dragMouseRel = QPoint(0, 0)
        
    def delete(self):
        if self.parent().dataAccessor():
            for decayObject in self.dataObjects():
                for decayObject2 in self.dataObjects():
                    if decayObject2.object() in self.parent().dataAccessor().motherRelations(decayObject.object()):
                        decayObject.object().unlinkMother(decayObject2.object())
                    if decayObject2.object() in self.parent().dataAccessor().daughterRelations(decayObject.object()):
                        decayObject.object().unlinkDaughter(decayObject2.object())                         
        return True
        
    def position(self, zoomed=False):
        if zoomed:
            return QPointF(self._position.x() * self.zoomFactor(), self._position.y() * self.zoomFactor())
        return self._position
    
    def setPosition(self, pos):
        self._position = pos
        
    def x(self):
        return self._position.x()
    
    def y(self):
        return self._position.y()
        
    def paint(self, painter, paintMode=0x0):
        if paintMode & DecayObject.PAINT_MODE_SELECTED:
            penColor = QColor(Qt.blue)
        #elif paintMode & DecayObject.PAINT_MODE_HOVERED:
        #    penColor = QColor(Qt.green)
        else:
            penColor = QColor(Qt.blue).lighter(140)
            if paintMode & DecayObject.PAINT_MODE_HOVERED:
                penColor = penColor.lighter(120)
        
        painter.setPen(QPen(penColor, 1 * self.zoomFactor(), Qt.SolidLine))
        painter.setBrush(penColor)
        #if paintMode & DecayObject.PAINT_MODE_HOVERED or paintMode & DecayObject.PAINT_MODE_SELECTED:
        painter.drawEllipse(self._position * self.zoomFactor(), self.CONTAINS_AREA_SIZE * 0.4 * self.zoomFactor(), self.CONTAINS_AREA_SIZE * 0.4 * self.zoomFactor())
        #painter.drawRect(self.boundingRect())
    
    def boundingRect(self):
        return self.containsAreaSquareRect(self._position)

    def containsPoint(self, pos):
        return self.boundingRect().contains(pos)
    
    def select(self, pos=None, selected=True):
        if pos:
            self._dragMouseRel = self._position - pos / self.zoomFactor()
        
    def move(self, *arg):
        
        if len(arg) == 1:
            pos = arg[0]
        if len(arg) > 1:
            pos = QPoint(arg[0], arg[1])
        
        if self.parent():
            oldBoundingRect = self.boundingRect()
            oldBoundingRects = {}
            for object in self.dataObjects():
                oldBoundingRects[object] = object.boundingRect()
            
        self._position = pos / self.zoomFactor() + self._dragMouseRel
            
        if self.parent():
            for object in self.dataObjects():
                self.parent().objectMoved(object, oldBoundingRects[object])
            self.parent().objectMoved(self, oldBoundingRect)
    
    def unite(self, node):
        #logging.debug(self.__class__.__name__ +": unite()")
        if node == self:
            return False
        
        useDataAccessor = False
        if self.parent().dataAccessor():
            useDataAccessor = True
            
        oldDecayObjects = self.dataObjects()[:]
        newDecayObjects = node.dataObjects()[:]
        for newDecayObject in newDecayObjects:
            self.appendObject(newDecayObject)
            self.parent().update(newDecayObject.boundingRect()) # old bounding rect
            nodeType = newDecayObject.replaceNode(node, self)
            self.parent().update(newDecayObject.boundingRect()) # new bounding rect
            if useDataAccessor:
                for oldDecayObject in oldDecayObjects:
                    if nodeType == DecayNode.TYPE_MOTHER and oldDecayObject.nodeType(self) == DecayNode.TYPE_DAUGHTER:
                        #newDecayObject.object().linkMother(oldDecayObject.object())
                        self.parent().dataAccessor().linkMother(newDecayObject.object(), oldDecayObject.object())
                    if nodeType == DecayNode.TYPE_DAUGHTER and oldDecayObject.nodeType(self) == DecayNode.TYPE_MOTHER:
                        #newDecayObject.object().linkDaughter(oldDecayObject.object())
                        self.parent().dataAccessor().linkDaughter(newDecayObject.object(), oldDecayObject.object())
        return True

    
class DecayLine(DecayObject):
    
    # new properties
    LINE_WIDTH = 2
    DEFAULT_LENGTH = 70
    LABEL_OFFSET = 4
    
    HUNDREDEIGHTY_OVER_PI = 180 / math.pi
    
    def __init__(self, parent, startPoint, endPoint):
        DecayObject.__init__(self, parent)
        self._pxlObject = None
        
        if isinstance(parent, LineDecayContainer):
            self._selfContained = False
            self._startNode = parent.addDecayNode(startPoint)
            self._endNode = parent.addDecayNode(endPoint)
        else:
            # make it possible to use DecayLine outside LineDecayContainer
            self._selfContained = True
            self._startNode = DecayNode(parent, startPoint)
            self._endNode = DecayNode(parent, endPoint)
        
        self._startNode.appendObject(self)
        self._endNode.appendObject(self)
        self._color = QColor(176, 179, 177)
        self._lineStyle = Qt.SolidLine
        self._label = None
        self._showLabel = True
        self._labelMatrix = None
        self._labelBoundingRect = None
        
    def delete(self):
        self._startNode.removeObject(self)
        self._endNode.removeObject(self)
        self.parent().removeObject(self._startNode)
        self.parent().removeObject(self._endNode)
        
        # remove this DecayObject's pxl particle from parent's pxl event / eventview
        self.parent().object().removeObject(self.object())
        return True
        
    def motherNode(self):
        return self._startNode
    
    def daughterNode(self):
        return self._endNode
        
    def setObject(self, object):
        self._pxlObject = object
        
    def object(self):
        return self._pxlObject
    
    def qtLineStyle(self):
        if not self.parent().dataAccessor():
            return Qt.SolidLine
        if self._lineStyle == self.parent().dataAccessor().LINE_STYLE_DASH:
            return Qt.DashLine
        elif self._lineStyle == self.parent().dataAccessor().LINE_STYLE_SOLID:
            return Qt.SolidLine
        return None
    
    def setLineStyle(self, style):
        self._lineStyle = style
        
    def setColor(self, color):
        self._color = color
        
    def setLabel(self, label):
        self._label = label
        
    def setShowLabel(self, show):
        self._showLabel = show
    
    def nodeType(self, node):
        if self._startNode == node:
            return DecayNode.TYPE_MOTHER
        if self._endNode == node:
            return DecayNode.TYPE_DAUGHTER
        return None
    
    def replaceNode(self, oldNode, newNode):
        if self._startNode == oldNode:
            self._startNode.removeObject(self)
            self._startNode = newNode
            self._startNode.appendObject(self)
        if self._endNode == oldNode:
            self._endNode.removeObject(self)
            self._endNode = newNode
            self._endNode.appendObject(self)
        return self.nodeType(newNode)
    
    def lineWidth(self):
        return self.LINE_WIDTH * self.zoomFactor()
    
    def dataAccessor(self):
        if self.parent():
            return self.parent().dataAccessor()
        return None
    
    def extendedSize(self):
        """ Returns True if instead of simple line a spiral or a sinus function is plotted.
        """
        if not self.parent().dataAccessor():
        #if not self.dataAccessor():
            return False
        return self._lineStyle == self.parent().dataAccessor().LINE_STYLE_SPIRAL or self._lineStyle == self.parent().dataAccessor().LINE_STYLE_WAVE
        
    def paint(self, painter, paintMode=0x0):
        if paintMode & DecayObject.PAINT_MODE_SELECTED:
            penColor = QColor(Qt.blue)
        else:
            penColor = self._color
            if paintMode & DecayObject.PAINT_MODE_HOVERED:
                penColor = penColor.lighter(150)
        
        if self.extendedSize():
            # spiral or wave line
            
            z = self.zoomFactor()
            if self._startNode.x() < self._endNode.x():
                xNull = self._startNode.x() * z
                yNull = self._startNode.y() * z
            else:
                xNull = self._endNode.x() * z
                yNull = self._endNode.y() * z
            #deltaX = abs(self._startNode.x() - self._endNode.x()) * z
            #deltaY = abs(self._startNode.y() - self._endNode.y()) * z
            
            l = self.length(zoomed = True)
            ## l = (n + 1/2) * r * 2 * math.pi
            designRadius = 1.2 * z
            # n: number of spirals
            n = max(int(1.0 * l / (2 * math.pi * designRadius)), 4)
            # r: radius of cycloide wheel
            r = 1.0 * l / (2 * math.pi * (n + 0.5))
            # a: cycloide is trace of point with radius a
            a = 3.5 * r
            
            path = QPainterPath()
            if self.parent().dataAccessor() and self._lineStyle == self.parent().dataAccessor().LINE_STYLE_SPIRAL:
                # draw spiral using a cycloide
                # coordinates of center of wheel
                xM = a
                yM = 0
                while xM < l:
                    phase = 1.0 * (xM-a)/r
                    x = xM - a * math.cos(phase)
                    y = yM - a * math.sin(phase)
                    if x > l:
                        x = l
                        #path.lineTo(QPointF(x, y))
                        break
                    path.lineTo(QPointF(x, y))
                    xM += 0.2 * r / z
            elif self.parent().dataAccessor() and self._lineStyle == self.parent().dataAccessor().LINE_STYLE_WAVE:
                x = a
                while x < l - 0.5*a:
                    y = a * math.cos(x/r)
                    path.lineTo(QPointF(x, y))
                    x += 0.2 * r / z
             
            slope = self.slope()
            angle = math.atan(slope)
            angleDegree = angle * self.HUNDREDEIGHTY_OVER_PI
            
            painter.translate(QPointF(xNull, yNull))    # rotate around start point
            painter.rotate(angleDegree)
            painter.setPen(QPen(penColor, 0.5*self.lineWidth(), Qt.SolidLine))
            painter.drawPath(path)
            painter.resetTransform()

        else:
            painter.setPen(QPen(penColor, self.lineWidth(), self.qtLineStyle()))
            painter.drawLine(self._startNode.position() * self.zoomFactor(), self._endNode.position() * self.zoomFactor())
            
        if not paintMode & DecayObject.PAINT_MODE_NO_DECORATIONS:
            if self.extendedSize():
                # don't recalculate angles
                self.drawText(painter, paintMode, slope, angle, angleDegree)
            else:
                self.drawText(painter, paintMode)
            
        if self._selfContained:
            self._startNode.paint(painter, paintMode)
            self._endNode.paint(painter, paintMode)
            
        #painter.drawRect(self.boundingRect())
    
    def drawText(self, painter, paintMode=0x0, *args):
        if not self._showLabel or not self._label:
            return
        if len(args) > 0:
            slope = args[0]
            #angle = args[1]
            angleDegree = args[2]
        else:
            slope = self.slope()
            #angle = math.atan(slope)
            angleDegree = math.atan(slope) * self.HUNDREDEIGHTY_OVER_PI
        
        font = QFont()
        font.setPointSize(12 * self.zoomFactor())
        fm = QFontMetrics(font)
        labelBoundingRect = fm.boundingRect(self._label)
        
        labelWidth = labelBoundingRect.width()
        if self._startNode.x() < self._endNode.x():
            startPoint = self._startNode.position(zoomed=True)
        else:
            startPoint = self._endNode.position(zoomed=True)
            
        label_offset = self.LABEL_OFFSET
        if self.extendedSize():
            label_offset += 2 
        offset = QPointF(0.5 * (self.length(zoomed=True) - labelWidth), - label_offset * self.zoomFactor())
        
        painter.translate(startPoint)    # rotate around start point
        painter.rotate(angleDegree)
        
        # for self.boundingRect()
        labelBoundingRect.translate(offset.x(), offset.y())
        self._labelMatrix=painter.combinedMatrix()
        self._labelBoundingRect = labelBoundingRect
        
        if paintMode & DecayObject.PAINT_MODE_SELECTED:
            textColor = QColor(Qt.blue)
        elif paintMode & DecayObject.PAINT_MODE_HOVERED:
            textColor = QColor(Qt.gray)
        else:
            textColor = QColor(Qt.black)

        path = QPainterPath()
        path.addText(offset, font, self._label)
        painter.fillPath(path, textColor)
        painter.resetTransform()
        
    def boundingRect(self):
        contains_area_size = self.CONTAINS_AREA_SIZE
        if self.extendedSize():
            contains_area_size += 4
            
        offset = contains_area_size * self.zoomFactor()
        startPoint = self._startNode.position() * self.zoomFactor()
        endPoint = self._endNode.position() * self.zoomFactor()
        
        topLeft = QPoint(min(startPoint.x(), endPoint.x()) - offset, min(startPoint.y(), endPoint.y()) - offset)
        bottomRight = QPoint(max(startPoint.x(), endPoint.x()) + offset, max(startPoint.y(), endPoint.y()) + offset) 
        
        rect = QRect(topLeft, bottomRight)
        if self._labelBoundingRect:
            return rect.united(self._labelMatrix.mapRect(self._labelBoundingRect))
        return rect
    
    def slope(self):
        deltaX = self._endNode.x() - self._startNode.x()
        if deltaX == 0:
            return sys.maxint
        
        return 1.0 * (self._endNode.y() - self._startNode.y()) / deltaX
    
    def length(self, zoomed=False):
        l = math.sqrt((self._endNode.x() - self._startNode.x())**2 + (self._endNode.y() - self._startNode.y())**2)
        if zoomed:
            return self.zoomFactor() * l
        return l
        
    def containsPoint(self, pos):
        pos = pos / self.zoomFactor()

        # label
        if self._labelBoundingRect and self._labelBoundingRect.contains(self._labelMatrix.inverted()[0].map(pos)):
            return True
        
        # line
        line_width = self.LINE_WIDTH + 1
        if self.extendedSize():
            line_width += 6
            
        if self._endNode.position().x() == self._startNode.position().x():
            # vertical
            if abs(pos.x() - self._endNode.position().x()) < line_width:
                return True
            return False
    
        if pos.x() < (min(self._startNode.position().x(), self._endNode.position().x()) - line_width) or pos.x() > (max(self._startNode.position().x(), self._endNode.position().x()) + line_width):
            return False
        
        slope = self.slope()
        deltaY = slope * (pos.x() - self._startNode.position().x()) + self._startNode.position().y() - pos.y()
        if abs(deltaY) < 0.5* line_width * max(1, abs(slope)):
            return True
        return False
        
    def select(self, pos=None, selected=True):
        if not pos:
            pos = (self._startNode.position() + self._endNode.position()) * 0.5
        self._startNode.select(pos)
        self._endNode.select(pos)
        
    def move(self, pos):
        self._startNode.move(pos)
        self._endNode.move(pos)

class ParticleWidget(VispaWidget):
    
    # inherited
    MINIMUM_WIDTH = 30
    MINIMUM_HEIGHT = 0
    
    # new
    NONE = 0
    LEPTON = 1
    QUARK = 2
    BOSON = 3
    HIGGS = 4
    
    def __init__(self, parent, type, name, dragData=None):
        self._dragStartPosition = None
        self._mimeDataType = None
        self._toolBoxContainer = None
        self._decayObect = None
        self._object = None

        VispaWidget.__init__(self, parent)
        self.enableAutosizing(True, False)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setSelectable(False)
        self.setDragable(False)
        self.setShape("ROUNDRECT")
        self.setText(name)
        self.textField().setPenColor(QColor("white"))
        self.show()
        if dragData:
            self._dragData = dragData
        else:
            self._dragData = name
            
        self.setMimeDataType(LineDecayView.DECAY_OBJECT_MIME_TYPE)
        
        if type == self.LEPTON:
            self.setColors(QColor(117, 57, 18), QColor(180, 88, 28), QColor(244, 119, 38))
        elif type == self.QUARK:
            self.setColors(QColor(19, 56, 0), QColor(27, 79, 27), QColor(57, 129, 51))
        elif type == self.BOSON:
            self.setColors(QColor(64, 0, 0), QColor(127, 0, 0), QColor(191, 0, 0))
        elif type == self.HIGGS:
            self.setColors(QColor(28, 63, 253), QColor(27, 118, 255), QColor(21, 169, 250))
        
    def setMimeDataType(self, type):
        self._mimeDataType = type
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragStartPosition = QPoint(event.pos())   # copy, does not work without on SL 4.0
        if isinstance(self.parent(), WidgetView):
            self.parent().widgetSelected(self)
            
    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        if (event.pos() - self._dragStartPosition).manhattanLength() < QApplication.startDragDistance():
            return
        
        drag = QDrag(self)
        mimeData = QMimeData()
        mimeData.setData(self._mimeDataType, self._dragData)
        drag.setMimeData(mimeData)
        drag.setPixmap(QPixmap.grabWidget(self))
        drag.setHotSpot(QPoint(drag.pixmap().width()/2, drag.pixmap().height()/2))
        drag.exec_()
        
    def setDecayObject(self, decayObject):
        """ Will be painted in content area.
        """
        self._decayObect = decayObject
        
    def setObject(self, object):
        """ The particle widget can optionally carry a real physics object, e.g. pxl particle.
        
        Required for example if widget's parent is a WidgetView that should react on clicks.
        """
        self._object = object
        
    def object(self):
        return self._object
        
    def paint(self, painter):
        VispaWidget.paint(self, painter)
        if self._decayObect:
            paintMode = 0
            if self.isSelected():
                paintMode |= DecayObject.PAINT_MODE_SELECTED
            self._decayObect.paint(painter, paintMode)
            
    def dataAccessor(self):
        """ Return None for decay object.
        """
        return None
