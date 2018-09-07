import sys
import math

from PyQt4.QtCore import Qt, QPoint, QPointF, QRect, QSize, SIGNAL, QCoreApplication, QMimeData, QRectF
from PyQt4.QtGui import QWidget, QPainter, QPolygon, QColor, QPen, QPalette, QPainterPath, QFont, QFontMetrics, QApplication, QDrag, QPixmap,QSizePolicy,QMessageBox, QTransform, QBrush

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
    import_autolayout_error=None
except Exception as e:
    import_autolayout_error=(str(e),exception_traceback())

class LineDecayView(WidgetView):
    
    LABEL = "&Line Decay View"

    DECAY_OBJECT_MIME_TYPE = "text/x-decay-object"
    WARNING_ABOVE = 1000
    
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
        self._noDecorationsMode=False
        self._topLevelContainer=None
        
        self._crateDecayObjectsDecaysThreadChain = ThreadChain()
        self.connect(self._crateDecayObjectsDecaysThreadChain, SIGNAL("finishedThreadChain"), self.createDecayObjectsThreadChainFinished)

    def noDecorationsMode(self):
        return self._noDecorationsMode
    
    def operationId(self):
        return self._operationId
        
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
        numObjects=self.numberDataObjectChildren()
        self._noDecorationsMode=numObjects>abs(self.WARNING_ABOVE)
        if self.WARNING_ABOVE>0 and numObjects>self.WARNING_ABOVE:
            result=QCoreApplication.instance().showMessageBox("You are about to display more than "+str(numObjects)+" (>"+str(self.WARNING_ABOVE)+") objects. This may take some time. Labels will not be displayed.",
                                                                       "Would you like to continue?",
                                                                       QMessageBox.Yes | QMessageBox.No,
                                                                       QMessageBox.Yes, [("Yes (remember my decision)",QMessageBox.YesRole)])
            if result == QMessageBox.No:
                self._updatingFlag -=1
                return False
            if result == 0:
                self.WARNING_ABOVE=-self.WARNING_ABOVE
        existingWidgets = []
        for object in self.applyFilter(self.dataObjects()):
            if object == None:
                # usually this should not happen, just in case dataAccessor misbehaves
                continue
            if self.dataAccessor().isContainer(object):
                # event or event view
                eventWidget = self.createLineDecayContainer(object)
                self._topLevelContainer=eventWidget
                existingWidgets += [eventWidget]
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
            if operationId != self.operationId():
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
        if import_autolayout_error!=None:
            logging.error(__name__ + ": Could not import autolayout algorithm: "+import_autolayout_error[1])
            QCoreApplication.instance().errorMessage("Could not import autolayout algorithm (see logfile for details):\n"+import_autolayout_error[0])
            return
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
        
# Apparently setTabController() is unnecessary, remove if sure about that (2010-06-29)
#    def setTabController(self, controller):
#        """ Sets tab controller.
#        """
#        WidgetView.setTabController(self, controller)
#        self.connect(self, SIGNAL("selected"), controller.onSelected)
        
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

    def topLevelContainer(self):
        return self._topLevelContainer
    
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
    
    def scrollBarValueChanged(self, event):
        for child in self.children():
            if isinstance(child, LineDecayContainer):
                child.scheduleUpdateVisibleList(True)
        
        
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
    
    def __init__(self, parent):
        logging.debug(self.__class__.__name__ +": __init__()")
        self._subWidgetStartY = 0
        self._updateVisibleListFlag = True
        ObjectHolder.__init__(self)
        WidgetContainer.__init__(self, parent)
        self.setEditable(parent.editable())
        self.setSelectable(True)
        
        self._selectedList = []
        self._visibleList = []
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

    def noDecorationsMode(self):
        return self.parent().noDecorationsMode()
            
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
                
        self.scheduleUpdateVisibleList()
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
        
    def createObject(self,dropType,pos):
        newObject = None
        if dropType == "Node":
            newObject = self.addDecayNode(pos)
            self.select(newObject)
            self.setFocus()
        elif dropType == "EventView":
            newObject = self.createChildContainer(self.object().createEventView(), pos)
            newObject.select(True)
            newObject.setFocus()
            # connect selected signal to parent
            parent=self
            while hasattr(parent,"parent"):
                if hasattr(parent,"onSelected"):
                    self.connect(newObject, SIGNAL("selected"), parent.onSelected)
                    break
                parent=parent.parent()
        else:
            # assume dropType is the generic name or type of a particle
            newObject = self.addParticleByType(dropType, pos)
            self.select(newObject)
            self.setFocus()
                
        if newObject:
            self.update(newObject.boundingRect())
            if self._editable:
                self.autosize()     # instead of full autolayout
            if self.tabController():
                self.tabController().setModified()
                if hasattr(self.tabController(),"updateTreeView"):
                    self.tabController().updateTreeView()
        return newObject
        
    def dropEvent(self, event):
        """ Handle drop of module.
        """
        logging.debug(self.__class__.__name__ + ": dropEvent()")
        if not self.tabController():
            return
        
        if event.mimeData().hasFormat(LineDecayView.DECAY_OBJECT_MIME_TYPE):
            dropType = str(event.mimeData().data(LineDecayView.DECAY_OBJECT_MIME_TYPE))
            pos = event.pos() / self.zoomFactor()
            if self.createObject(dropType, pos):
                event.acceptProposedAction()
                    
    def addParticleByType(self, particleType=None, pos=None):
        """ This function asks the data accessor to create a new particle and makes sure it gets the properties (name, pdg id, charge) of the desired type.
        
        particleType may either be None or a string specifying the object type.
        pos is in 100% coordinates.
        """
        
        if not particleType:
            # this is nothing
            return None
        dataAccessor = self.dataAccessor()
        
        newParticle = dataAccessor.createParticle()
        #newParticle.setName(particleType)  # use this name to find id
        categoryName = "Object info"
        dataAccessor.setProperty(newParticle, "Name", particleType, categoryName)  # use this name to find id
        particleId = self.dataAccessor().particleId(newParticle)
        if particleId != None:
            #newParticle.setPdgNumber(particleId)
            dataAccessor.setProperty(newParticle, "PdgNumber", particleId, categoryName)
        particleName = self.dataAccessor().defaultName(newParticle)
        if particleName:
            # normalize name using id
            #newParticle.setName(particleName)
            dataAccessor.setProperty(newParticle, "Name", particleName, categoryName)
        #newParticle.setCharge(self.dataAccessor().charge(newParticle))
        dataAccessor.setProperty(newParticle, "Charge", dataAccessor.charge(newParticle), categoryName) #TODO: check whether this is necessary
        self._pxlObject.setObject(newParticle)
        
        return self.addDecayLine(newParticle, pos)
    
    def addDecayLine(self, object, pos=None):
        """ This function accepts a data object (e.g. a pxl object), creates a DecayLine for it and adds the latter to this container's objects list. 
        
        """
        if not object:
            return None
        
        if not pos:
            pos = QPoint(10, 10)
        else:
            pos -= QPoint(0.5 * DecayLine.DEFAULT_LENGTH, 0)
        
        # id() is slow, so ids of existing objects are stored
        if self.dataAccessor():
            dataObjectId = self.dataAccessor().id(object)
            if dataObjectId in self._existingObjectIds:
                # maybe here should the existing object be returned
                return None                
            self._existingObjectIds.append(str(dataObjectId))
        
        # map pxl relations to gui components
        # find potentially, already existing DecayNodes to be used by the DecayLine that will be created below
        motherNode = None
        daughterNode = None 
        if self.dataAccessor():
            for daughter in self.dataAccessor().daughterRelations(object):
                if daughter in self._particlesDict.keys():
                    daughterDecayObject = self._particlesDict[daughter]
                    if not daughterNode:
                        daughterNode = daughterDecayObject.motherNode()
                    elif daughterDecayObject.motherNode() != daughterNode and \
                        daughterDecayObject.motherNode().unite(daughterNode):
                            # already found one daughter
                            # need to make sure all of the other relevant nodes are united
                            self.removeObject(daughterNode)
                            daughterNode = daughterDecayObject.motherNode()
                                
            for mother in self.dataAccessor().motherRelations(object):
                if mother in self._particlesDict.keys():
                    motherDecayObject = self._particlesDict[mother]
                    if not motherNode:
                        motherNode = motherDecayObject.daughterNode()
                    elif motherDecayObject.daughterNode() != motherNode and \
                        motherDecayObject.daughterNode().unite(motherNode):
                            # already found one mother
                            # need to make sure all of the other relevant nodes are united
                            self.removeObject(motherNode)
                            motherNode = motherDecayObject.daughterNode()
        if not motherNode:
            motherNode = QPoint(pos)
        if not daughterNode:
            daughterNode = QPoint(pos.x() + DecayLine.DEFAULT_LENGTH, pos.y())
            
        # create DecayLine
        newDecayLine = DecayLine(self, motherNode, daughterNode)
        newDecayLine.setObject(object)
        if self.dataAccessor():
            newDecayLine.setLabel(self.dataAccessor().label(object))
            newDecayLine.setColor(self.dataAccessor().color(object))
            newDecayLine.setLineStyle(self.dataAccessor().lineStyle(object))
        self._particlesDict[object] = newDecayLine

        self.appendObject(newDecayLine)    
        self.scheduleUpdateVisibleList()
        return newDecayLine
        
    def addDecayNode(self, pos):
        newDecayNode = DecayNode(self, pos)
        self.scheduleUpdateVisibleList()
        return self.appendObject(newDecayNode)
    
    def operationId(self):
        return self.parent().operationId()
    
    def createDecayObjectsFromPxlObject(self, operationId):
        """ Creates DecayObjects for all particles in the set pxl object.
        
        In addition this function is called on all child LineDecayContainers.
        """
        #logging.debug(self.__class__.__name__ +": createDecayObjectsFromPxlObject()")
        if self._pxlObject and self.dataAccessor():
            for childObject in self.applyFilter(self.dataAccessor().children(self._pxlObject)):
                if operationId != self.operationId():
                    return False
                if self.dataAccessor().isContainer(childObject):
                    self.createChildContainer(childObject)
                else:
                    self.addDecayLine(childObject)
                
        for child in self.children():
            #if not Application.NO_PROCESS_EVENTS:
            #    QCoreApplication.instance().processEvents()
            if operationId != self.operationId():
                return False
            if isinstance(child, LineDecayContainer):
                if not child.createDecayObjectsFromPxlObject(operationId):
                    return False
        return True
            
    def decayObject(self, pxlObject):
        """ Returns the DecayObject which represents the given pxlObject or None if there is no such one.
        
        This function is to slow for massive usage with many dataObjects as it loops over all dataObjects.
        """
        for decayObject in self.dataObjects():
            if decayObject.object() == pxlObject:
                return decayObject
        return None
    
    def select(self, decayObject):
        if isinstance(decayObject, type(True)):
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
        
    def objectMoved(self, decayObject, oldBoundingRect=None):
        boundingRect = decayObject.boundingRect()
        if oldBoundingRect:
            self.update(boundingRect.unite(oldBoundingRect))
        else:
            self.update(boundingRect)
        
        # update visiblity list
        objectIsVisible = False
        if self.visibleRegion().intersects(boundingRect):
            objectIsVisible = True
        if decayObject in self._visibleList:
            if not objectIsVisible:
                self._visibleList.remove(decayObject)
        elif objectIsVisible:
            self._visibleList.append(decayObject)
        
    def scheduleUpdateVisibleList(self, update=True):
        self._updateVisibleListFlag = update
        for child in self.children():
            if isinstance(child, LineDecayContainer):
                child.scheduleUpdateVisibleList(update)

    def updateVisibleList(self, force=False):
        if not self._updateVisibleListFlag and not force:
            return
        
        #logging.debug("%s: updateVisibleList()" % self.__class__.__name__)
        
        region = self.visibleRegion()
        self._visibleList = []
        for decayObject in reversed(self.dataObjects()):
        #for decayObject in oldVisibleList + self.dataObjects():
            if region.intersects(decayObject.boundingRect()):
                self._visibleList.append(decayObject)
        
        self._updateVisibleListFlag = False
        
    def showEvent(self, event):
        self.scheduleUpdateVisibleList()
        
    def paint(self, painter):
        WidgetContainer.paint(self, painter)
        
        if self.collapsed():
            # don't paint if container is collapsed
            return
        
        generalPaintMode = 0x0
        if self.noDecorationsMode():
            generalPaintMode = DecayObject.PAINT_MODE_NO_DECORATIONS
        
        #if self.dataObjectsCount() > 50:
        #    painter.setRenderHint(QPainter.Antialiasing, False)
        
        self.updateVisibleList()
        decayNodes = []
        painterClipRegion = self.visibleRegion()
        for decayObject in reversed(self._visibleList):
            if isinstance(decayObject, DecayLine):
                self.paintDecayObject(painter, decayObject, generalPaintMode)
            if isinstance(decayObject, DecayNode):
                # paint nodes after lines, so that they appear above the lines
                decayNodes.append(decayObject)
                    
        for decayObject in decayNodes:
            self.paintDecayObject(painter, decayObject, generalPaintMode)
                
    def paintDecayObject(self, painter, decayObject, generalPaintMode):
        paintMode = generalPaintMode
        if decayObject in self._selectedList:
            paintMode |= DecayObject.PAINT_MODE_SELECTED
        if decayObject == self._hoveredObject:
            paintMode |= DecayObject.PAINT_MODE_HOVERED
        decayObject.paint(painter, paintMode)
        
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
        if self._titleClicked:
            if self._editable:
                WidgetContainer.mouseMoveEvent(self, event)
            return
        elif self.menu():
            if self.isTitlePoint(event.pos()):
                self.positionizeMenuWidget()
                self.showMenu()
            else:
                self.menu().hide()
            
        if not bool(event.buttons()):
            # no button pressed -> hovering
            to_hover_object = None
            if not self._visibleList:
                self._visibleList = self.dataObjects()
            
            for object in self._visibleList:
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
            controller=self.tabController()
            if controller:
                controller.tab().propertyView().clear()
                
            if self.isSelected():
                self.delete()
            elif len(self._selectedList) > 0:
                self.removeObject(self._selectedList[0])
            
            if controller:
                controller.setModified()
                if hasattr(controller,"updateTreeView"):
                    controller.updateTreeView()
                    #controller.autolayout()
    
    def delete(self):
        parent = self.parent()
        if WidgetContainer.delete(self) and hasattr(parent, "object"):
            parent.object().removeObject(self.object())
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
            self.scheduleUpdateVisibleList()
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
            if i == 0 or self._nodeVector[i].position.y > maxY:
                maxY = self._nodeVector[i].position.y
                
            if (len(self._nodeVector[i].mothers) == 1 and len(self._nodeVector[i].children) == 0) or (len(self._nodeVector[i].mothers) == 0 and len(self._nodeVector[i].children) == 1):
                # orphan particles:
                # nodes have exactly one relation
                # eighter node is mother or daughter (left and right side of particle)
                if firstMinOrphanY or self._nodeVector[i].position.y > minOrphanY:
                    minOrphanY = self._nodeVector[i].position.y
                    firstMinOrphanY = False
            else:
                # non orphans
                if firstMaxNonOrphanY or self._nodeVector[i].position.y < maxNonOrphanY:
                    maxNonOrphanY = self._nodeVector[i].position.y
                    firstMaxNonOrphanY = False

        xOffset = -30
        yOffset = -minY + (self.getDistance("titleFieldBottom") + 4* self.getDistance("topMargin")) / self.zoomFactor()
        
        for decayNode in self.dataObjects():
            if isinstance(decayNode, DecayNode) and decayNode in self._allNodes.keys():
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
    LABEL_OFFSET = 7
    
    ARROW_LENGTH = 14   # length of two small arrow lines (painted when selected or hovered)
    ARROW_WIDTH = 8     # vertical distance of two arrow lines from normal line
    
    HUNDREDEIGHTY_OVER_PI = 180 / math.pi
    
    def __init__(self, parent, startPointOrNode, endPointOrNode):
        self._color = QColor(176, 179, 177)
        self._lineStyle = Qt.SolidLine
        self._label = None
        self._labelFont = None
        self._showLabel = True
        self._labelMatrix = None
        self._labelBoundingRect = None
        self._recalculateBoundingRect = True
        self._boundingRect = None
        self._arrowBoundingRect = None
        self._forwardDirection = True
        self._recalculateTransform = True
        self._transform = None
        self._pxlObject = None
        
        DecayObject.__init__(self, parent)
        
        if isinstance(parent, LineDecayContainer):
            self._selfContained = False
            if isinstance(startPointOrNode, QPoint):
                # startPoint
                self._startNode = parent.addDecayNode(startPointOrNode)
            else:
                # startNode
                self._startNode = startPointOrNode
            if isinstance(endPointOrNode, QPoint):
                # endPoint
                self._endNode = parent.addDecayNode(endPointOrNode)
            else:
                # endNode
                self._endNode = endPointOrNode
        else:
            # make it possible to use DecayLine outside LineDecayContainer
            self._selfContained = True
            self._startNode = DecayNode(parent, startPointOrNode)
            self._endNode = DecayNode(parent, endPointOrNode)
        
        self._startNode.appendObject(self)
        self._endNode.appendObject(self)
        
    def setZoom(self, zoom):
        DecayObject.setZoom(self, zoom)
        if self._labelFont:
            self._labelFont.setPointSize(12 * self.zoomFactor())
        
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
        self._recalculateBoundingRect = True
        
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
        self._recalculateBoundingRect = True
        
    def setColor(self, color):
        self._color = color
        self._recalculateBoundingRect = True
        
    def setLabel(self, label):
        self._label = label
        if not self._labelFont:
            self._labelFont = QFont()
            self._labelFont.setPointSize(12 * self.zoomFactor())
        
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
        self._recalculateBoundingRect = True
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
            return False
        return self._lineStyle == self.parent().dataAccessor().LINE_STYLE_SPIRAL or self._lineStyle == self.parent().dataAccessor().LINE_STYLE_WAVE or self._lineStyle == self.parent().dataAccessor().LINE_VERTEX
        
    def transform(self):
        """ Returns QTransform that sets the origin to the start point and rotates by the slope angle.

        Used to change coordinates of painter in paint().
        """

        if not self._recalculateTransform and self._transform:
            return self._transform
        
        z = self.zoomFactor()
        if self._startNode.x() < self._endNode.x():
            self._forwardDirection = True
            xNull = self._startNode.x() * z
            yNull = self._startNode.y() * z
        else:
            self._forwardDirection = False
            xNull = self._endNode.x() * z
            yNull = self._endNode.y() * z
        
        slope = self.slope()
        angle = math.atan(slope)
        angleDegree = angle * self.HUNDREDEIGHTY_OVER_PI
        
        self._transform = QTransform()
        self._transform.translate(xNull, yNull)    # rotate around start point
        self._transform.rotate(angleDegree)
        return self._transform
        
    def paint(self, painter, paintMode=0x0):
        if paintMode & DecayObject.PAINT_MODE_SELECTED:
            penColor = QColor(Qt.blue)
        else:
            penColor = self._color
            if paintMode & DecayObject.PAINT_MODE_HOVERED:
                penColor = penColor.lighter(80)
                
        showDirectionArrow = paintMode & DecayObject.PAINT_MODE_HOVERED or paintMode & DecayObject.PAINT_MODE_SELECTED
        extendedSize = self.extendedSize()
        
        z = self.zoomFactor()
        l = self.length(zoomed = True)
        
        # transform coordinates to make following calculations easier
        painter.setTransform(self.transform())
        
        if extendedSize:
            painter.setPen(QPen(penColor, 0.5*self.lineWidth(), Qt.SolidLine))
            # spiral or wave line
            ## l = (n + 1/2) * r * 2 * math.pi
            designRadius = 1.2 * z
            # n: number of spirals
            n = max(int(1.0 * l / (2 * math.pi * designRadius)), 4)
            # r: radius of cycloide wheel
            r = 1.0 * l / (2 * math.pi * (n + 0.5))
            # a: cycloide is trace of point with radius a
            a = 3.5 * r
            
            if self.parent().dataAccessor() and self._lineStyle == self.parent().dataAccessor().LINE_STYLE_SPIRAL:
                path = QPainterPath()
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
                painter.drawPath(path)
            elif self.parent().dataAccessor() and self._lineStyle == self.parent().dataAccessor().LINE_STYLE_WAVE:
                path = QPainterPath()
                x = a
                while x < l - 0.5*a:
                    y = a * math.cos(x/r)
                    path.lineTo(QPointF(x, y))
                    x += 0.2 * r / z
                painter.drawPath(path)
            elif self.parent().dataAccessor() and self._lineStyle == self.parent().dataAccessor().LINE_VERTEX:
                painter.setBrush(QBrush(penColor, Qt.SolidPattern))
                painter.drawEllipse(QPointF(l/2.,0.),l/2.,a)

        else:
            painter.setPen(QPen(penColor, self.lineWidth(), self.qtLineStyle()))
            painter.drawLine(QPoint(0,0), QPoint(l, 0))
            
        if not paintMode & DecayObject.PAINT_MODE_NO_DECORATIONS:
            self.drawText(painter, paintMode)
            
        # paint arrow lines
        if showDirectionArrow:
            painter.setPen(QPen(penColor, 0.8*self.lineWidth(), Qt.SolidLine, Qt.RoundCap))
            self.drawArrow(painter, paintMode)
        
        ## DEBUG
        #painter.setPen(QPen(penColor, 1, Qt.SolidLine))
        #painter.drawRect(self.labelBoundingRect())
        #painter.drawRect(self.arrowBoundingRect())
        
        painter.resetTransform()
        #painter.drawRect(self.boundingRect())
        
        if self._selfContained:
            self._startNode.paint(painter, paintMode)
            self._endNode.paint(painter, paintMode)
            
    # simple paint method for debugging
#    def paint(self, painter, paintMode=0x0):
#        if paintMode & DecayObject.PAINT_MODE_SELECTED:
#            penColor = QColor(Qt.blue)
#        else:
#            penColor = self._color
#            if paintMode & DecayObject.PAINT_MODE_HOVERED:
#                penColor = penColor.lighter(150)
#       
#        painter.setPen(QPen(penColor, self.lineWidth(), Qt.SolidLine))
#        painter.drawLine(self._startNode.position() * self.zoomFactor(), self._endNode.position() * self.zoomFactor())
#            
#        if not paintMode & DecayObject.PAINT_MODE_NO_DECORATIONS:
#            self.drawText(painter, paintMode)
#            
#        if self._selfContained:
#            self._startNode.paint(painter, paintMode)
#            self._endNode.paint(painter, paintMode)
#            
#        #painter.drawRect(self.boundingRect())
    
    def drawText(self, painter, paintMode=0x0):
        """ Draws self._label on given painter.
        
        Expects coordinates of painter transformed as returned by transform()
        """
        if not self._showLabel or not self._label:
            return
        
        if paintMode & DecayObject.PAINT_MODE_SELECTED:
            textColor = QColor(Qt.blue)
        elif paintMode & DecayObject.PAINT_MODE_HOVERED:
            textColor = QColor(Qt.gray)
        else:
            textColor = QColor(Qt.black)

        path = QPainterPath()
        path.addText(QPointF(self.labelBoundingRect().bottomLeft()), self._labelFont, self._label)
        painter.fillPath(path, textColor)
        
    def drawArrow(self, painter, paintMode=0x0):
        # make sure to stay within bounding rect, therefore add / substract arrowPixelOffset
        arrowPixelOffset =  0.7* self.lineWidth()
        arrowBoundingRect = self.arrowBoundingRect()
        arrowBoundingRectLeft = arrowBoundingRect.left() + arrowPixelOffset
        arrowBoundingRectRight = arrowBoundingRect.right() - arrowPixelOffset
        arrowBoundingRectTop = arrowBoundingRect.top() + arrowPixelOffset
        arrowBoundingRectBottom = arrowBoundingRect.bottom() - arrowPixelOffset
        arrowBoundingRectVerticalCenter = arrowBoundingRect.center().y()
        if self._forwardDirection:
            painter.drawLine(arrowBoundingRectLeft, arrowBoundingRectTop, arrowBoundingRectRight, arrowBoundingRectVerticalCenter)
            painter.drawLine(arrowBoundingRectLeft, arrowBoundingRectBottom, arrowBoundingRectRight, arrowBoundingRectVerticalCenter)
            
        else:
            painter.drawLine(arrowBoundingRectLeft, arrowBoundingRectVerticalCenter, arrowBoundingRectRight, arrowBoundingRectTop)
            painter.drawLine(arrowBoundingRectLeft, arrowBoundingRectVerticalCenter, arrowBoundingRectRight, arrowBoundingRectBottom)
        
        
    def labelBoundingRect(self, forceRecalculation=False):
        if not self._label or not self._labelFont:
            return QRect()
        
        if not self._labelBoundingRect or forceRecalculation:
            label_offset = self.LABEL_OFFSET
            if self.extendedSize():
                label_offset += 2 
            
            fm = QFontMetrics(self._labelFont)
            self._labelBoundingRect = fm.boundingRect(self._label)
            
            labelWidth = self._labelBoundingRect.width()
            offset = QPointF(0.5 * (self.length(zoomed=True) - labelWidth), - label_offset * self.zoomFactor())
            self._labelBoundingRect.translate(offset.x(), offset.y())
            
        return self._labelBoundingRect
    
    def arrowBoundingRect(self, forceRecalculation=False):
        if not self._arrowBoundingRect or forceRecalculation:
            zoomFactor = self.zoomFactor()
            l = self.length(zoomed = True)
            arrowLength = self.ARROW_LENGTH * zoomFactor
            arrowWidth = self.ARROW_WIDTH * zoomFactor
            horizontalOffset = self.CONTAINS_AREA_SIZE * 0.4 * self.zoomFactor()    # offset from end of line
            if self._forwardDirection:
                self._arrowBoundingRect = QRect(l-arrowLength - horizontalOffset, -arrowWidth, arrowLength, 2*arrowWidth)
            else:
                self._arrowBoundingRect = QRect(horizontalOffset, -arrowWidth, arrowLength, 2*arrowWidth)
        
        return self._arrowBoundingRect
    
    def boundingRect(self):
        if not self._recalculateBoundingRect and self._boundingRect:
            return self._boundingRect
        
        self._recalculateTransform = True
        contains_area_size = self.CONTAINS_AREA_SIZE
        if self.extendedSize():
            contains_area_size += 4
            
        zoomFactor = self.zoomFactor()
        offset = contains_area_size * zoomFactor
        startPoint = self._startNode.position() * zoomFactor
        endPoint = self._endNode.position() * zoomFactor
        
        topLeft = QPoint(min(startPoint.x(), endPoint.x()) - offset, min(startPoint.y(), endPoint.y()) - offset)
        bottomRight = QPoint(max(startPoint.x(), endPoint.x()) + offset, max(startPoint.y(), endPoint.y()) + offset) 
        
        rect = QRect(topLeft, bottomRight)
        
        # increase rect for label and arrow (shows when selected or hovered
        rect = rect.united(self.transform().mapRect(self.arrowBoundingRect(True)))
        if self._label and self._labelFont:
            rect = rect.united(self.transform().mapRect(self.labelBoundingRect(True)))

        self._boundingRect = rect
        return self._boundingRect
    
    def slope(self):
        deltaX = self._endNode.x() - self._startNode.x()
        if deltaX == 0:
            return sys.maxsize
        
        return 1.0 * (self._endNode.y() - self._startNode.y()) / deltaX
    
    def length(self, zoomed=False):
        l = math.sqrt((self._endNode.x() - self._startNode.x())**2 + (self._endNode.y() - self._startNode.y())**2)
        if zoomed:
            return self.zoomFactor() * l
        return l
        
    def containsPoint(self, pos):
        pos = pos / self.zoomFactor()

        # label
        if self._label and self._labelFont and self.labelBoundingRect().contains(self.transform().inverted()[0].map(pos)):
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
        self._recalculateBoundingRect = True
        
    def move(self, pos):
        self._startNode.move(pos)
        self._endNode.move(pos)
        self._recalculateBoundingRect = True

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
            
        if hasattr(parent,"particleDoubleClicked"):
            self.connect(self,SIGNAL("mouseDoubleClicked"),parent.particleDoubleClicked)
        if hasattr(parent,"particleRightClicked"):
            self.connect(self,SIGNAL("mouseRightPressed"),parent.particleRightClicked)
        
    def setMimeDataType(self, type):
        self._mimeDataType = type

    def dragData(self):
        return self._dragData

    def mouseDoubleClickEvent(self, event):
        self.emit(SIGNAL("mouseDoubleClicked"), self)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragStartPosition = QPoint(event.pos())   # copy, does not work without on SL 4.0
        if isinstance(self.parent(), WidgetView):
            self.parent().widgetSelected(self)
        if event.button()==Qt.RightButton:
            self.emit(SIGNAL("mouseRightPressed"), event.globalPos(), self)
            
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

 	  	 
