import logging

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Vispa.Main.VispaWidget import *
from Vispa.Main.VispaWidgetOwner import *
from Vispa.Main.PortConnection import *

class WidgetContainer(VispaWidget, VispaWidgetOwner):
    
    BACKGROUND_SHAPE = 'ROUNDRECT'
    #PEN_COLOR = QColor('blue')
    PEN_COLOR = QColor(0, 75, 141)
    FILL_COLOR1 = QColor('white')
    FILL_COLOR2 = QColor('white')
    
    WIDTH = 10
    HEIGHT = 10
        
    AUTORESIZE = True
    AUTORESIZE_KEEP_ASPECT_RATIO = False
    
    def __init__(self, parent=None):
        #logging.debug(__name__ + ": __init__()")
        self._childrenVisible = True
        VispaWidget.__init__(self, parent)
    
    def defineBackgroundBrush(self):
        """ If container is collapsed use single background color.
        """
        if self._childrenVisible:
            VispaWidget.defineBackgroundBrush(self)
        else:
            self._backgroundBrush = self.penColor()
        
    def sizeHint(self):
        """ Calculates needed space for content.
        """
        #logging.debug(__name__ +": sizeHint()")
        superClassNeededSpace = VispaWidget.sizeHint(self)
        neededWidth = superClassNeededSpace.width()
        neededHeight = superClassNeededSpace.height()
        
        childrenRect = self.childrenRect()
        width = childrenRect.bottomRight().x() + self.getDistance('rightMargin')
        height = childrenRect.bottomRight().y() + self.getDistance('bottomMargin')
        
        minWidth = WidgetContainer.WIDTH
        minHeight = WidgetContainer.HEIGHT
        
        if width < minWidth:
            width = minWidth
        if height < minHeight:
             height = minHeight
                
        if width < superClassNeededSpace.width():
            width = superClassNeededSpace.width()
        if height < superClassNeededSpace.height():
            height = superClassNeededSpace.height()
        
        # make zoom independent
        width = 100.0 / self.zoom() * width
        height = 100.0 / self.zoom() * height
        
        return QSize(width, height)
    
    def autosize(self):
        """Sets size of child widget to the size needed to fit whole content. 
        """
        #logging.debug(__name__ +": autosize() "+ str(self.title()))
        if not self.autoresizeEnabled():
            return
        
        childrenRect = self.childrenRect()
        #print 'childrenRect', childrenRect
       
        # adjust size and position left and top
        xOffset = - childrenRect.x()
        yOffset = - childrenRect.y()
        # min offsets effectively define margins
        xMinOffset = self.getDistance('leftMargin')
        yMinOffset = self.getDistance('topMargin') + self.getDistance('titleFieldBottom')
        
        if self._childrenVisible:
            # Prevent 'jumping' when collapsing / uncollapsing
            if xOffset <= xMinOffset:
                xOffset += xMinOffset
            if yOffset <= yMinOffset:
                yOffset += yMinOffset
        
        xOffset = round(xOffset)
        yOffset = round(yOffset)
                
        self.move(self.x() - xOffset , self.y() - yOffset)
        for child in self.children():
            child.move(child.x() + xOffset, child.y() + yOffset)
        
        #self.scheduleRearangeContent()
        #self.scaleChanged()
        VispaWidget.autoresize(self)
        self.update()
        
        # Forward change information to parent
        if isinstance(self.parent(), WidgetContainer):
            self.parent().autosize()
        self.emit(SIGNAL("sizeChanged"), self)
        
    def widgetMoved(self, widget):
        """ Call autosize().
        
        Overwritten function of VispaWidgetOwner.
        """
        VispaWidgetOwner.widgetMoved(self, widget)
        self.autosize()
        for connection in [child for child in self.children() if isinstance(child, PortConnection)]:
            connection.updateConnection()
    
    def mouseDoubleClickEvent(self, event):
        """ Call toggleCollapse().
        """
        self.toggleCollapse()
        
    def mousePressEvent(self, event):
        """ Makes sure event is forwarded to both base classes.
        """
        VispaWidgetOwner.mousePressEvent(self, event)
        VispaWidget.mousePressEvent(self, event)
        
    def toggleCollapse(self):
        """ Toggles visibility of children between True and False.
        """
        if self._childrenVisible:
            self._childrenVisible = False
        else:
            self._childrenVisible = True
            
        for child in self.children():
            child.setVisible(self._childrenVisible)
        
        self.setTitleBackgroundColorEnabled(self._childrenVisible)
        self.autosize()
        self.widgetMoved(self)
