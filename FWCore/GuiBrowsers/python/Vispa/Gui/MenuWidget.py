from PyQt4.QtCore import Qt, QSize, SIGNAL, QEvent, QPointF
from PyQt4.QtGui import QColor, QPen, QBrush, QRadialGradient

import logging

from Vispa.Gui.VispaWidget import VispaWidget, TextField

class MenuWidget(VispaWidget):
    
    # inherited properties
    HEIGHT = 30
    BACKGROUND_SHAPE = 'ROUNDRECT'
    ROUNDRECT_RADIUS = 10

    PEN_COLOR = QColor(242, 230, 242)
    FILL_COLOR1 = QColor(59, 59, 59, 200)
    #FILL_COLOR2 = QColor(242, 230, 242)
    TITLE_COLOR = QColor(Qt.white)
    
    AUTOSIZE = True
    AUTOSIZE_KEEP_ASPECT_RATIO = False
    SELECTABLE_FLAG = False
    USE_BACKGROUND_GRADIENT_FLAG = False
    
    # new properties
    HOVER_COLOR1 = QColor(0, 0, 240)
    HOVER_COLOR2 = QColor(0, 0, 200)
    
    def __init__(self, parent=None, associatedWidget=None, orientation=Qt.Horizontal):
        """ Constructor
        """
        logging.debug("%s: Constructor" % self.__class__.__name__)
        self._cursorEntered = False
        
        self._menuEntryTextFields = []
        self._menuEntrySlots = []
        self._hoverEntry = None
        
        self._spacer = TextField()
        
        VispaWidget.__init__(self, parent)
        self.hide()
        self._associatedWidget = associatedWidget
        self.setMouseTracking(True)
        self.setDragable(False)
        #self._hoverBrush = QBrush(self.HOVER_COLOR1)
        self._hoverGradient = QRadialGradient()
        self._hoverGradient.setColorAt(0, self.HOVER_COLOR1)
        self._hoverGradient.setColorAt(1, self.HOVER_COLOR2)
        #self._hoverBrush = QBrush(self.HOVER_COLOR1)
        
        self._spacer.setFontSizeRange(self.TEXTFIELD_FONTSIZE_MIN, self.TEXTFIELD_FONTSIZE_MAX)
            #self._textField.setDefaultWidth(self.getDistance('textFieldWidth', 1, True))
        #entry.setDefaultHeight(self.getDistance('textFieldHeight', 1, True))
        self._spacer.setDefaultFontSize(self.TEXTFIELD_FONTSIZE)
        self._spacer.setAutosizeFont(self.TEXTFIELD_AUTOSIZE_FONT_FLAG)
        self._spacer.setAutotruncate(self.TEXTFIELD_AUTOTRUNCATE_TEXT_FLAG)
        self._spacer.setAutoscale(True, False)
        self._spacer.setPenColor(self.TITLE_COLOR)
        self._spacer.setFont(self.font())
        self._spacer.setText(" | ")
        self._spacer.calculateDimensions()
        
    def addEntry(self, name, slot=None):
        entry = TextField()
        entry.setFontSizeRange(self.TEXTFIELD_FONTSIZE_MIN, self.TEXTFIELD_FONTSIZE_MAX)
            #self._textField.setDefaultWidth(self.getDistance('textFieldWidth', 1, True))
        #entry.setDefaultHeight(self.getDistance('textFieldHeight', 1, True))
        entry.setDefaultFontSize(self.TEXTFIELD_FONTSIZE)
        entry.setAutosizeFont(self.TEXTFIELD_AUTOSIZE_FONT_FLAG)
        entry.setAutotruncate(self.TEXTFIELD_AUTOTRUNCATE_TEXT_FLAG)
        entry.setAutoscale(True, False)
        entry.setPenColor(self.TITLE_COLOR)
        entry.setFont(self.font())
        entry.setText(name)
        entry.calculateDimensions()
        self._menuEntryTextFields.append(entry)
        self._menuEntrySlots.append(slot)
        
        self.scheduleRearangeContent()
        return entry
    
    def removeEntry(self, entry):
        if entry in self._menuEntryTextFields:
            index = self._menuEntryTextFields.index(entry)
            self._menuEntryTextFields.remove(entry)
            self._menuEntrySlots.pop(index)
            
    def setEntryText(self, entry, text):
        if entry in self._menuEntryTextFields:
            entry.setText(text)
            entry.calculateDimensions()
            self.scheduleRearangeContent()
    
    def len(self):
        return len(self._menuEntryTextFields)
    
    def entry(self, index):
        if len(self._menuEntryTextFields) >= index + 1:
            return self._menuEntryTextFields[index]
        return None
            
    def sizeHint(self):
        """ Calculates needed space for widget content.
        """
        self._scaleWidth = 1         # for getDistance()
        self._scaleHeight = 1
        
        neededWidth = self.getDistance('leftMargin', 1) + self.getDistance('rightMargin', 1)
        neededHeight = self.getDistance('topMargin', 1) + self.getDistance('bottomMargin', 1)
        
        textFieldWidth = 0
        textFieldHeight = 0
        for entry in self._menuEntryTextFields:
            textFieldWidth += entry.getWidth()
            textFieldHeight = max(textFieldHeight, entry.getHeight())
        textFieldWidth += max(0, (len(self._menuEntryTextFields) -1) * self._spacer.getWidth())
                
        neededWidth += textFieldWidth
        neededHeight += textFieldHeight
        
        # evaluate maximum size
        maxWidth = self.maximumSize().width()
        maxHeight = self.maximumSize().height()
        
        maxScaleWidth = min(1.0, 1.0 * maxWidth/neededWidth)
        maxScaleHeight = min(1.0, 1.0 * maxHeight/neededHeight)
        if maxScaleWidth != 1.0 or maxScaleHeight != 1.0:
            # this is not limited by keepAspectRationFlag
            # as it is about absolute sizes here
            # ratio is evaluated in autosize()
            scale = min(maxScaleWidth, maxScaleHeight)
            neededWidth *= scale
            neededHeight *= scale
        
        return QSize(max(self.minimumSize().width(), neededWidth), max(self.minimumSize().height(), neededHeight))
    
    def paint(self, painter, event=None):
        """ Takes care of painting widget content on given painter.
        """
        VispaWidget.paint(self, painter, event)
        self.drawMenuEntries(painter)
        
    def drawMenuEntries(self, painter):
        """ Tells TextField object of text field to draw title on widget.
        """
        x = self.getDistance('textFieldX')
        y = self.getDistance('textFieldY')
        spacerWidth = self._spacer.getWidth() * self.zoomFactor()
        originalPen = QPen()
        painter.setPen(originalPen)
        firstFlag = True
        for entry in self._menuEntryTextFields:
            if not firstFlag:
                self._spacer.paint(painter, x, y, self.zoomFactor())
                x += spacerWidth
            if self._hoverEntry == entry:
                hoverRectXOffset = 0.3 * spacerWidth
                hoverRectYOffset = 0.3 * self.getDistance('topMargin')
                hoverWidth = entry.getWidth() * self.zoomFactor() + 2 * hoverRectXOffset
                hoverHeight = entry.getHeight() * self.zoomFactor() + 2 * hoverRectYOffset
                #self._hoverGradient.setColorAt(0, self.HOVER_COLOR1)
                #self._hoverGradient.setColorAt(1, self.HOVER_COLOR2)
                self._hoverGradient.setCenter(QPointF(entry.getDrawRect(self.zoomFactor()).center()))
                self._hoverGradient.setFocalPoint(QPointF(entry.getDrawRect(self.zoomFactor()).center()))
                self._hoverGradient.setRadius(min(hoverWidth, hoverHeight))
                #painter.setBrush(self._hoverBrush)
                painter.setBrush(self._hoverGradient)
                painter.setPen(Qt.NoPen)
                #painter.drawRoundedRect(entry.getDrawRect(self.zoomFactor()), 10, 10)
                painter.drawRoundedRect(x - hoverRectXOffset, y - hoverRectYOffset, hoverWidth, hoverHeight, 5, 5)
                painter.setPen(originalPen)
            entry.paint(painter, x, y, self.zoomFactor())
            x += entry.getWidth() * self.zoomFactor()
            firstFlag = False
    
    def mouseMoveEvent(self, event):
        if bool(event.buttons() & Qt.LeftButton):
            VispaWidget.mouseMoveEvent(self, event)
            return
        for entry in self._menuEntryTextFields:
            if entry.getDrawRect(self.zoomFactor()).contains(event.pos()):
                self._hoverEntry = entry
                self.update()
                break
            
    def mousePressEvent(self, event):
        VispaWidget.mousePressEvent(self, event)
        for i, entry in enumerate(self._menuEntryTextFields):
            if self._menuEntrySlots[i] and entry.getDrawRect(self.zoomFactor()).contains(event.pos()):
                self.hide()
                self._menuEntrySlots[i]()
                break
            
    def leaveEvent(self, event):
        #logging.debug("%s: leaveEvent()" % self.__class__.__name__)
        if not self._associatedWidget or self.parent().childAt(self.parent().mapFromGlobal(self.cursor().pos())) != self._associatedWidget:
            self.hide()
        self._hoverEntry = None
        self.update()
        
    def hide(self):
        VispaWidget.hide(self)
        self._hoverEntry = None
    
    def showEvent(self, event):
        VispaWidget.showEvent(self, event)
        self._cursorEntered = False

    def enterEvent(self, event):
        self._cursorEntered = True
        
    def cursorHasEntered(self):
        return self._cursorEntered
