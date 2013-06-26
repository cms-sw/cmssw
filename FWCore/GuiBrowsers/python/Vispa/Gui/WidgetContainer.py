import logging

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Vispa.Gui.VispaWidget import VispaWidget
from Vispa.Gui.ConnectableWidget import ConnectableWidget
from Vispa.Gui.ConnectableWidgetOwner import ConnectableWidgetOwner
from Vispa.Gui.PortConnection import PortConnection

class WidgetContainer(ConnectableWidget, ConnectableWidgetOwner):
    
    # inherited properties
    BACKGROUND_SHAPE = 'ROUNDRECT'
    #PEN_COLOR = QColor('blue')
    PEN_COLOR = QColor(0, 75, 141)
    FILL_COLOR1 = QColor('white')
    FILL_COLOR2 = QColor('white')
    
    WIDTH = 10
    HEIGHT = 10
        
    AUTOSIZE = True
    AUTOSIZE_KEEP_ASPECT_RATIO = False
    
    # new properties
    AUTOLAYOUT_CHILDREN_ENABLED = False
    AUTOSIZE_ADJUST_CONTAINER_POSITION = True

    def __init__(self, parent=None):
        #logging.debug(__name__ + ": __init__()")
        self._childrenVisible = True
        self._autolayoutChildrenEnabled = False
        self._autosizeAdjustContainerPositionFlag = True
        self._collapseMenuButton = None
        self._collapsable=True
        self._hiddenChildren = []
        
        ConnectableWidget.__init__(self, parent)
        
        self.enableAutolayoutChildren(self.AUTOLAYOUT_CHILDREN_ENABLED)
        self.enableAdjustContainerPositionWhenAutosizing(self.AUTOSIZE_ADJUST_CONTAINER_POSITION)
        
        self.setShowCollapseMenu(True)
        
    def enableAutolayoutChildren(self, autolayout):
        """ If autolayout is True children of this container are arranged when this container auto-sizes.
        
        If this option is enabled make sure children are not dragable because that would conflict with  autolayouting.
        """
        self._autolayoutChildrenEnabled = autolayout
        
    def enableAdjustContainerPositionWhenAutosizing(self, adjust):
        self._autosizeAdjustContainerPositionFlag = adjust
        
    def adjustContainerPositionWhenAutosizingEnabled(self):
        return self._autosizeAdjustContainerPositionFlag
        
    def autolayoutChildrenEnabled(self):
        return self._autolayoutChildrenEnabled
    
    def defineBackgroundBrush(self):
        """ If container is collapsed use single background color.
        """
        if self._childrenVisible:
            ConnectableWidget.defineBackgroundBrush(self)
        else:
            self._backgroundBrush = self.penColor()
        
    def sizeHint(self):
        """ Calculates needed space for content.
        """
        #logging.debug(__name__ +": sizeHint()")
        childrenRect = self.childrenRect()
        width = childrenRect.bottomRight().x() + self.getDistance('rightMargin')
        height = childrenRect.bottomRight().y() + self.getDistance('bottomMargin')
        
        # make zoom independent
        width = 1.0 / self.zoomFactor() * width
        height = 1.0 / self.zoomFactor() * height
        
        # from now on in 100% dimensions
        superClassNeededSpace = VispaWidget.sizeHint(self)
        
        width = max(width, superClassNeededSpace.width(), WidgetContainer.WIDTH)
        height = max(height, superClassNeededSpace.height(), WidgetContainer.HEIGHT)
        
        return QSize(width, height)

    def contentStartX(self):
        """ Get start x coordinate position of widget content. Especially for autolayoutChildren().
        """
        return self.getDistance("contentStartX")
    
    def contentStartY(self):
        """ Get start y coordinate position of widget content. Especially for autolayoutChildren().
        """
        return self.getDistance("contentStartY")

    def autosize(self):
        """ Overwrite VispaWidget's function.
        
        This adds size and position handling for left and top sides of container.
        VispaWidget only takes care of right and bottom sides as widget position remains constant.
        This function also adjusts the container's position as needed to include all children.
        """
        if self._autolayoutChildrenEnabled and self._childrenVisible:
            self.autolayoutChildren()
            self.updateConnections()
            
        childrenRect = WidgetContainer.childrenRect(self)
        if self._autosizeAdjustContainerPositionFlag and self._childrenVisible and (childrenRect.width() != 0 or childrenRect.height() != 0):
            # increase / decrease size of container on left and top side
            
            # round to prevent drifting wehen zoom != 100%
            xMargin = round(self.contentStartX())
            yMargin = round(self.contentStartY())
            childrenRectX = childrenRect.x()
            childrenRectY = childrenRect.y() 
            
            xOffset = 0
            yOffset = 0
            if childrenRectX != xMargin:
                xOffset = - childrenRectX + xMargin
            if childrenRectY != yMargin:
                yOffset = - childrenRectY + yMargin
                
            if xOffset != 0 or yOffset != 0:
                self.move(self.x() - xOffset , self.y() - yOffset)
                for child in self.children():
                    #if isinstance(child,QWidget): # needed for PyQt4.5
                    if hasattr(child, "move"):
                        child.move(child.x() + xOffset, child.y() + yOffset)
        
        VispaWidget.autosize(self)
        self.emit(SIGNAL("sizeChanged"), self)
        
    def autolayoutChildren(self):
        """ This function arranges children one after each other in a column.
        
        See setAuotlayoutChildrenEnabled().
        """
        #logging.debug(self.__class__.__name__ +": autolayoutChildren()")
        
        # round to prevent drifting when zoom != 100%
        xPos = round(self.contentStartX())
        yPos = round(self.contentStartY())
        
        for child in self.children():
            if isinstance(child, VispaWidget) and child.isVisible():
                child.move(xPos, yPos)
                yPos += child.height() + self.getDistance("topMargin")
                
    def widgetDragged(self, widget):
        """ Call autosize().
        
        Overwritten function of ConnectableWidgetOwner.
        """
        if self.autosizeEnabled():
            self.autosize()
        # not sure this is still needed (2010-07-06), remove if possible
        #for connection in [child for child in self.children() if isinstance(child, PortConnection)]:
        #    connection.updateConnection()
            
        ConnectableWidgetOwner.widgetDragged(self, widget)
    
    def mouseDoubleClickEvent(self, event):
        """ Call toggleCollapse().
        """
        if event.pos().y() <= self.getDistance("titleFieldBottom"):
            self.toggleCollapse()
        if isinstance(self.parent(), ConnectableWidgetOwner):
            self.parent().widgetDoubleClicked(self)
        
    def mousePressEvent(self, event):
        """ Makes sure event is forwarded to both base classes.
        """
        ConnectableWidgetOwner.mousePressEvent(self, event)
        VispaWidget.mousePressEvent(self, event)
        
    def mouseMoveEvent(self, event):
        if bool(event.buttons() & Qt.LeftButton):
            VispaWidget.mouseMoveEvent(self, event)
        elif self.menu():
            self.positionizeMenuWidget()
        
        if event.pos().y() <= self.getDistance("titleFieldBottom"):
            self.showMenu()
        elif self.menu():
            self.menu().hide()
        
    def collapsed(self):
        """ Returns True if widget is collapsed. In this case the children are not visible.
        
        Otherwise False is returned.
        """
        return not self._childrenVisible
        
    def toggleCollapse(self):
        """ Toggles visibility of children between True and False.
        """
        if self.menu():
            self.menu().hide()
            
        
        if self._childrenVisible:
            self._hiddenChildren = []
            self._childrenVisible = False
        else:
            self._childrenVisible = True
            
        for child in self.children():
            if isinstance(child,QWidget): # needed for PyQt4.5
                if not self._childrenVisible and not child.isVisible():
                    # remember already hidden children while hiding
                    self._hiddenChildren.append(child)
                elif not child in self._hiddenChildren:
                    # prevent to make previously hidden children visible
                    child.setVisible(self._childrenVisible)
        
        if self._childrenVisible:
            self.enableBackgroundGradient(self.setColors(self.PEN_COLOR, self.FILL_COLOR1, self.FILL_COLOR2))
            self.enableColorHeaderBackground(self.COLOR_HEADER_BACKGROUND_FLAG)
            self.setColors(self.PEN_COLOR, self.FILL_COLOR1, self.FILL_COLOR2)
        else:
            self.enableBackgroundGradient(False)
            self.enableColorHeaderBackground(False)
            self.setColors(self.PEN_COLOR, self.PEN_COLOR, None)
            
        self.autosize()
        self.widgetDragged(self)
        
    def showMenu(self):
        if self._collapseMenuButton:
            if self._childrenVisible and self._collapsable:
                self.menu().setEntryText(self._collapseMenuButton, "Collapse")
                #self._collapseMenuButton.setText("Collapse")
            else:
                self.menu().setEntryText(self._collapseMenuButton, "Expand")
                #self._collapseMenuButton.setText("Expand")
        ConnectableWidget.showMenu(self)
        
    def setShowCollapseMenu(self, show=True):
        if show and not self._collapseMenuButton:
            self._collapseMenuButton = self.addMenuEntry("", self.toggleCollapse)
        elif not show and self._collapseMenuButton:
            self.removeMenuEntry(self._collapseMenuButton)
            self._collapseMenuButton = None

    def setNotCollapsable(self):
        self._collapsable=False
        