import math
    
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtSvg import QSvgRenderer

import logging

from Vispa.Gui.ZoomableWidget import ZoomableWidget
from Vispa.Gui.VispaWidgetOwner import VispaWidgetOwner

class TextField(object):
    """ TextField for VispaWidget.
    
    Text and title shown in VispaWidget are TextField object.
    """
    
    WIDTH = 100
    HEIGHT = 0           # If set to zero, it will be automatically set to the font's height in setFont().
    
    FONT_SIZE = 12
    
    def __init__(self):
        self._text = ''
        self._textShort = ''
        self._font = None             #needed for autosizeFont()
        self._fontSize = self.FONT_SIZE
        self._fontSizeHasChanged = True
        self._penColor = QColor(Qt.black)
        self._minFontSize = 1
        self._maxFontSize = 30
        self._outputFlags = Qt.AlignLeft
        
        self._defaultWidth = self.WIDTH
        self._defaultHeight = self.HEIGHT
        self._width = self._defaultWidth
        self._height = self._defaultHeight
    
        self._deletableFalg = True
        self._autosizeFontFlag = False
        self._autotruncateTextFlag = True
        self._autoscaleFlag = False
        self._autoscaleKeepAspectRatioFlag = True
        
        self._xPos = 0
        self._yPos = 0
    
    def setText(self, text):
        """ Sets text.
        """
        self._text = text
        self._textShort = ''
    
    def setAutosizeFont(self, auto):
        """ Sets autosizeFontFlag.
        
        If flag is True and text does not fit in its given area the font size will be reduced to fit.
        """
        self._autosizeFontFlag = bool(auto)
        if not self.empty():
            self.calculateDimensions()
    
    def setAutotruncate(self, auto):
        """ Sets autoTruncateTextFlag.
        
        If flag is True the text will be truncated if it is too long to fit in given space.
        """
        self._autotruncateTextFlag = bool(auto)
        if not self.empty():
            self.calculateDimensions()
    
    def setAutoscale(self, auto, keepAspectRatio):
        """ Sets autoscale and autoscalKeepAspectRatio flags.
        
        If autoscale flag is True the needed space is increased until text fits.
        If keepAspectRatio flag is False the aspet ratio may change depending on output flags.
        See setOutputFlags().
        """
        self._autoscaleFlag = auto
        self._autoscaleKeepAspectRatioFlag = keepAspectRatio
        
    def setFont(self, qfont):
        """ Sets font and if default height is not yet set default height will be set to font height.
        """
        #self._font = QFont(qfont)
        self._font = qfont
        self._fontSizeHasChanged = True
            
    def font(self):
        return self._font
    
    def setPenColor(self, color):
        self._penColor = color
        
    def penColor(self):
        return self._penColor
    
    def getFontHeight(self, fm=None):
        """ Calculates font height for given font metrics object.
        
        If no font metrics object is given one will be created for the current TextField font.
        """
        if fm == None:
            fm = QFontMetrics(self._font)
        height = fm.height()
        return height
    
    def setDefaultFontSize(self, fontSize):
        """ Sets preferred font size.
        """
        self._defaultFontSize = fontSize
    
    def setFontSizeRange(self, minFontSize, maxFontSize):
        """ Sets min and max font point size for autosize font capability.
        
        See setAutosizeFont().
        """
        self._minFontSize = minFontSize
        self._maxFontSize = maxFontSize
    
    def getDrawRect(self, scale=1):
        """ Returns QRect.
        
        Width will be equal to getWidth() and height equal to getHeight().
        """
        return QRect(self._xPos, self._yPos, math.ceil(self._width * scale), math.ceil(self._height * scale))
        #return QRectF(self._xPos, self._yPos, self._width, self._height)

    def setDefaultWidth(self, width):
        """ Sets preferred width for text output.
        """
        self._defaultWidth = width
    
    def setDefaultHeight(self, height):
        """ Sets preferred height for text output.
        """
        self._defaultHeight = height
    
    def calculateDimensions(self):
        """ Calculates the space (width and height) needed to display text.
        
        Depending on the flags set the size will be greater than the default size,
        or the font size will be adjusted,
        or the text will be truncated.
        
        See setAutosizeFont(), setAutotruncate(), setAutoscale(), setDefaultWidth(), setDefaultHeight().
        """
        #self._width = self._defaultWidth
        #self._height = self._defaultHeight
        
        if self._fontSizeHasChanged and (not self._autosizeFontFlag or self._autoscaleFlag):
            self._font.setPointSize(self.getFontSize())
            if self._defaultHeight == 0:
                self.setDefaultHeight(self.getFontHeight())
            self._fontSizeHasChanged = False
        
        if self._autoscaleFlag:
            self.autoscale()
            
        elif self._autosizeFontFlag:
            self.autosizeFont()
        
        if self._autotruncateTextFlag:
            self.truncate()

    def getWidth(self):
        """ Returns width calculated by calculateDimensions().
        """
        return self._width
    
    def getHeight(self):
        """ Returns height calculated by calculateDimensions().
        """ 
        #logging.debug(self.__class__.__name__ + ": getHeight() "+ str(self._height) + " " + self.text())
        return self._height
        
    def setOutputFlags(self, flags):
        """ Set Qt output flags for drawing text.
        """
        self._outputFlags = flags
    
    def  getOutputFlags(self):
        """ Returns set output flags.
        """
        return self._outputFlags
    
    def truncated(self):
        """ Returns True if text was truncated.
        """
        if self._textShort != '':
            return True
        return False
    
    def getFontSize(self):
        """ Returns the font size the text will be drawn in.
        """
        if self._autoscaleFlag:
            return self._defaultFontSize
        return self._fontSize
    
    def text(self):
        """ Returns text.
        """
        return self._text
    
    def getTextShort(self):
        """ Returns short version of text if it was truncated.
        """
        return self._textShort
    
    def getOutputText(self):
        """ Evaluates whether the string was truncated or not.
        
        If truncated it returns short version, else the whole text.
        """
        if self.truncated():
            return self._textShort
        return self._text
    
    def empty(self):
        """ Returns True if no text or empty string is set.
        """
        if self._text == '' or self._text == None:
            return True
        return False
    
    def autoscale(self):
        """ Adjusts values for getWidth() and getHeight() so whole text fits in.
        """
        #logging.debug("%s: autoscale() - %s" % (self.__class__.__name__, self._text))
        fm = QFontMetrics(self._font)
        self.ranbefore=True
        self._width = 1
        self._height = 1
        widthFits = heightFits = False
        
        if not self._autoscaleKeepAspectRatioFlag:
            # test for replacing else part in while-loop (2009-02-23)
            neededRect = fm.boundingRect(0, 0, self._defaultWidth*100, self._defaultHeight*100, self._outputFlags, self._text)
            self._width = neededRect.width()
            self._height = neededRect.height()
            return
        
        while not widthFits or not heightFits:
            if self._autoscaleKeepAspectRatioFlag:
                self._width += 1
                self._height = 1.0 * self._width * (self._defaultHeight + 1) / self._defaultWidth
                # 'defaultHeight' +1 prevents factor 0 --> infinite loop
            else:
                if not widthFits:
                    self._width += 1
                if not heightFits:
                    self._height += 1
            neededRect = fm.boundingRect(0, 0, self._width, self._height, self._outputFlags, self._text)
            if neededRect.width() <= self._width:
                widthFits = True
                self._width += 1 # prevent slightly too small width (maybe due to rounding while zoooming)
            if neededRect.height() <= self._height:
                heightFits = True
        #logging.debug(self.__class__.__name__ +": autoscale() - (width, height) = ("+ str(self._width) +", "+ str(self._height) +")")
        
    def autosizeFont(self):
        """ Decreases font so text fits in given widht and height.
        """
        if self._font == None:
            logging.error("TextField.autosizeFont() - ERROR: 'font' not set, can't calculate font size")
            return
            
        drawRect = self.getDrawRect()
        font = self._font
        decSize = 0
        for size in range(self._minFontSize + 1, self._maxFontSize + 1):
            font.setPointSizeF(size + 0.1 * decSize)    
            fm = QFontMetricsF(font)
            neededRect = fm.boundingRect(drawRect, self._outputFlags, self._text)
            if neededRect.width() > drawRect.width() or neededRect.height() > drawRect.height():
                size -= 1
                break
            
        for decSize in range(0, 10):
            font.setPointSizeF(size + 0.1 * decSize)    
            fm = QFontMetricsF(font)
            neededRect = fm.boundingRect(drawRect, self._outputFlags, self._text)
            if neededRect.width() > drawRect.width() or neededRect.height() > drawRect.height():
                decSize -= 1
                break
            
        self._fontSize = size + 0.1 * decSize
        #print "determineTextFieldSize(", self._fontSize, ")"
        
    def truncate(self):
        """ Truncates text if it does not fit in given space.
        """
        #logging.debug(self.__class__.__name__ + ": truncate()")
        text = QString(self._text)
        short = QString()
        drawRect = QRectF(self.getDrawRect())
        font = self._font
        fm = QFontMetricsF(font)
        counter = 0
        patterns = text.split(QRegExp('\\b'))
        
        for pattern in patterns:
            short.append(pattern)
            neededRect = fm.boundingRect(drawRect, self._outputFlags, short)
            
            if neededRect.width() > drawRect.width() or neededRect.height() > drawRect.height():
                break
            counter += len(pattern)
        
        if counter < len(text):
            self._textShort = text.left(counter)
            self._textShort = text.left(counter).append("...")
            #print "truncate() - short: ", self._textShort
        
    def paint(self, painter, xPos, yPos, scale=1):
        """ Draws text on given painter at given position.
        
        If scale is given the text will be scaled accordingly.
        """
        self._xPos = xPos
        self._yPos = yPos
        drawRect = self.getDrawRect(scale)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(self._penColor))
        self._font.setPointSize(max(self.getFontSize() * scale, 1))
        painter.setFont(self._font)
        painter.drawText(drawRect, self.getOutputFlags(), self.getOutputText())
        
        ## debug
        #painter.drawRect(drawRect)
        #print "   drawRect ", drawRect
        #print "   text", self.getOutputText()
        #print "drawRect(width, height) = ", drawRect.width(), ",", drawRect.height(), ")"


class VispaWidget(ZoomableWidget):
    """ Class supporting random shapes, title and text field.
    
    Title and text field content are stored in TextField objects.
    You can influence the behavior of the widget by severals flags.
    If you want the widget's size to fit the content size (increases / decreases widget) use enableAutosizing().
    You can also chose the decrease the text field's font size, if text does not fit in. See setTextFieldAutosizeFont().
    Additionally the text can be truncated instead using setTextFieldAutotruncateText().   
    """
    
    WIDTH = 100
    HEIGHT = 80
    MINIMUM_WIDTH = 30
    MINIMUM_HEIGHT = 0
    ROUNDRECT_RADIUS = 30
    BACKGROUND_SHAPE = 'RECT'
    
    TOP_MARGIN = 5
    LEFT_MARGIN = 5
    BOTTOM_MARGIN = 5
    RIGHT_MARGIN = 5
    HORIZONTAL_INNER_MARGIN = 5
    VERTICAL_INNTER_MARGIN = 5
    
    #PEN_COLOR = QColor('darkolivegreen')
    #FILL_COLOR1 = QColor('yellowgreen')
    #FILL_COLOR2 = QColor('darkkhaki')
    
    #PEN_COLOR = QColor(0, 116, 217)         # strong blue, end wprkshop
    #FILL_COLOR1 = QColor(65, 146, 217)
    #FILL_COLOR2 = QColor(122, 186, 242)
    
    #PEN_COLOR = QColor(102, 133, 176)        # gentle blue
    #FILL_COLOR1 = QColor(128, 186, 224)
    #FILL_COLOR2 = QColor(188, 215, 241)
    
    #PEN_COLOR = QColor(115, 115, 115)        # grey
    #FILL_COLOR1 = QColor(166, 166, 166)
    #FILL_COLOR2 = QColor(217, 217, 217)
    
    PEN_COLOR = QColor(128, 186, 224)       # gentle blue right
    FILL_COLOR1 = QColor(188, 215, 241)
    FILL_COLOR2 = QColor(242, 230, 242)
    TITLE_COLOR = QColor(Qt.white)
    
    #PEN_COLOR = QColor(100, 133, 156)
    #FILL_COLOR1 = QColor(116, 155, 181)
    #FILL_COLOR2 = QColor(124, 166, 194)
    
    SELECT_COLOR = QColor('darkblue')
    
    SELECTABLE_FLAG = True
    FOCUSPOLICY = Qt.StrongFocus
    SELECTED_FRAME_WIDTH = 2            # Width in pixels of colored (SELECT_CORLOR) frame, when widget is in focus
    
    AUTOPOSITIONIZE_WHEN_ZOOMING_FLAG = True
    
    TITLEFIELD_FONTSIZE = 12
    COLOR_HEADER_BACKGROUND_FLAG = True
    USE_BACKGROUND_GRADIENT_FLAG = True
    
    TEXTFIELD_FONTSIZE = 12
    TEXTFIELD_FONTSIZE_MIN = 2
    TEXTFIELD_FONTSIZE_MAX = 20
    TEXTFIELD_FLAGS = Qt.TextWordWrap
    TEXTFIELD_AUTOSIZE_FONT_FLAG = False
    TEXTFIELD_AUTOTRUNCATE_TEXT_FLAG = False
    
    AUTOSIZE = False
    AUTOSIZE_KEEP_ASPECT_RATIO = True
    
    ARROW_SHAPE = None
    ARROW_SHAPE_TOP = 0
    ARROW_SHAPE_LEFT = 1
    ARROW_SHAPE_BOTTOM = 2
    ARROW_SHAPE_RIGHT = 3
    ARROW_SIZE = 30
    
    def __init__(self, parent=None):
        """ Constructor
        """
        #print "VispaWidget.__init__()"
        self._autosizeFlag = False
        self._autosizeKeepAspectRatioFlag = True
        self._titleField = None
        self._colorHeaderBackgroundFlag = True
        self._backgroundGradientEnabledFlag = True

        self._scale = 1
        self._scaleWidth = 1
        self._scaleHeight = 1
        
        self.framePenColor = None
        self.fillColor1 = None
        self.fillColor2 = None
    
        self._textField = None    
        self._selectableFlag = False
        self._selectedFlag = False
        self._deletableFlag = True
        self._dragableFlag = True
        self._dragMouseXrel = 0
        self._dragMouseYrel = 0 
        
        self._distances = None
        self._distancesHaveToBeRecalculatedFlag = True
        self._rearangeContentFlag = True
        self._noRearangeContentFlag = True
        self._distancesLastScale = 0
        self._distancesLastScaleWidth = 0
        self._distancesLastScaleHeight = 0
    
        self._backgroundShape = ''
        self._backgroundShapePath = None
        self._arrowShape = None
    
        self._unzoomedPositionX = 0     # These values are set by the overridden move() function.
        self._unzoomedPositionY = 0     # With these values the widgets position can be scaled when zooming.
        self._autopositioninzeWhenZoomingFlag = False
        
        self._bodyWidget = None
        self._image = None
    
        ZoomableWidget.__init__(self, parent)
        
        self.setColors(self.PEN_COLOR, self.FILL_COLOR1, self.FILL_COLOR2)
        self.setSelectable(self.SELECTABLE_FLAG)
        self.setShape(self.BACKGROUND_SHAPE)
        self.setArrowShape(self.ARROW_SHAPE)
        self.enableAutopositionizeWhenZooming(self.AUTOPOSITIONIZE_WHEN_ZOOMING_FLAG)
        self.enableAutosizing(self.AUTOSIZE, self.AUTOSIZE_KEEP_ASPECT_RATIO)
        self.enableColorHeaderBackground(self.COLOR_HEADER_BACKGROUND_FLAG)
        self.enableBackgroundGradient(self.USE_BACKGROUND_GRADIENT_FLAG)
        self.setMinimumSize(self.MINIMUM_WIDTH, self.MINIMUM_HEIGHT)
        
        self.noRearangeContent(False)
        self.scheduleRearangeContent()
        self._previusDragPosition = self.pos()
        
    def unzoomedX(self):
        """ Returns x coordinate the widget would have if zoom was set to 100%.
        """
        #logging.debug(self.__class__.__name__ +": unzoomedX() "+ str(self._unzoomedPositionY))
        return int(self._unzoomedPositionX)
        
    def unzoomedY(self):
        """ Returns x coordinate the widget would have if zoom was set to 100%.
        """
        #logging.debug(self.__class__.__name__ +": unzoomedY() "+ str(self._unzoomedPositionY))
        return int(self._unzoomedPositionY)
        
    def scale(self):
        """ Return scale factor of widget.
        """
        return self._scale
    
    def setZoom(self, zoom):
        """ Sets widget's zoom.
        """
        ZoomableWidget.setZoom(self, zoom)
        
        self._scale = self.zoom() * 0.01
        ZoomableWidget.resize(self, self.width(), self.height())
        
        if self._autopositioninzeWhenZoomingFlag:
            self.move(self._unzoomedPositionX * self.scale(), self._unzoomedPositionY * self.scale())
            
    def penColor(self):
        """ Returns pen color for this widget.
        """
        return self.framePenColor
    
    def enableAutopositionizeWhenZooming(self, auto):
        """ If True the position of this widget will be corrected according to it unzoomed position.
        
        Prevents unexpected moving when zoom is enabled due to rounding errors.
        """
        self._autopositioninzeWhenZoomingFlag = bool(auto)
    
    def setDragable(self, dragable, recursive=False):
        """ If True the widget can be dragged using a pointing device.
        
        If recursive is True also dragablitiy of all children will be set. 
        """
        self._dragableFlag = dragable
        if recursive:
            for child in self.children():
                if isinstance(child, VispaWidget):
                    child.setDragable(dragable, True)
    
    def isDragable(self):
        return self._dragableFlag
    
    def setColors(self, penColor, fillColor1, fillColor2):
        """ Sets colors of this widget.
        
        The pen color will be used for drawing the widget's frame.
        Fill color 1 and 2 will be used for background color gradient.
        """
        self.framePenColor = penColor
        self.fillColor1 = fillColor1
        self.fillColor2 = fillColor2    
    
    def setShape(self, shape):
        """ Sets shape of this widget.
        
        Right now supported 'RECT' (default), 'ROUNDRECT', 'CIRCLE'.
        """
        self._backgroundShape = shape
        
    def setArrowShape(self, direction):
        """ In addition to normal shape this gives whole widget the shape of an arrow.
        
        Possible values for direction are
        None
        ARROW_SHAPE_TOP
        ARROW_SHAPE_LEFT
        ARROW_SHAPE_BOTTOM
        ARROW_SHAPE_RIGHT
        """
        self._arrowShape = direction
    
    def enableAutosizing(self, auto, keepAspectRatio=True):
        """ Sets flag for auto resizing this widget.
        
        If auto is True the size of this widget will be adjusted to widget size.
        If keepAspectRatio is False the aspect ratio may change according to widget content.
        """
        if self._autosizeKeepAspectRatioFlag != auto or self._autosizeKeepAspectRatioFlag != keepAspectRatio:
            changed = True
        else:
            changed = False
            
        self._autosizeFlag = auto
        self._autosizeKeepAspectRatioFlag = keepAspectRatio
        if self.titleIsSet():
            self._titleField.setAutoscale(auto, keepAspectRatio)
        if self.textFieldIsSet():
            self._textField.setAutoscale(auto, keepAspectRatio)
        
        if changed:
            self.scheduleRearangeContent()
            self.update()
            
    def autosizeEnabled(self):
        """ Returns True if auto resizing is enabled.
        
        See enableAutosizing().
        """
        return self._autosizeFlag
    
    def setSelectable(self, selectable):
        """ Makes widget selectable if True.
        """
        self._selectableFlag = bool(selectable)
        
        if self._selectableFlag:
            self.setFocusPolicy(self.FOCUSPOLICY)
        else:
            self.setFocusPolicy(Qt.NoFocus)
    
    def isSelectable(self):
        """ Returns True if widget can be selected.
        """
        return self._selectableFlag
        
    def setDeletable(self, deleteable):
        self._deletableFlag = deleteable
    
    def isDeletable(self):
        return self._deletableFlag
    
    def enableColorHeaderBackground(self, enable=True):
        """ If set to True the background of the header is painted using pen color.
        
        See setColors().
        """
        self._colorHeaderBackgroundFlag = enable
    
    def colorHeaderBackgroundEnabled(self):
        return self._colorHeaderBackgroundFlag
    
    def enableBackgroundGradient(self, enable=True):
        """ If set to True the background color is painted using a QLinearGradient with the two fill colors.
        
        See setColors().
        """
        self._backgroundGradientEnabledFlag = enable
        
    def isUseBackgroundGradientEnabled(self):
        return self._backgroundGradientEnabledFlag
    
    def select(self, sel=True, multiSelect=False):
        """ Marks this widget as selected and informs parent if parent is VispaWidetOwner.
        
        If multiSelect is True e.g. due to a pressed Ctrl button while clicking (see mousePressEvent()),
        the VispaWidgetOwner will also be informed and might not deselect all widgets.
        """
        if not self.isSelectable():
            return
        
        if sel != self._selectedFlag:
            self.update()
        
        self._selectedFlag = sel
        
        # TODO: raise in front of other widget, not in front of line
#        if sel:
#            self.raise_()

        if (multiSelect or self.isSelected()) and isinstance(self.parent(), VispaWidgetOwner):
            self.parent().widgetSelected(self, multiSelect)
    
    def mouseDoubleClickEvent(self, event):
        if isinstance(self.parent(), VispaWidgetOwner):
            self.parent().widgetDoubleClicked(self)
    
    def isSelected(self):
        """ Returns True if widget is selected.
        """
        if not self.isSelectable():
            return False
        
        return self._selectedFlag
        
    def _initTitleField(self):
        """ Initializes title field.
        
        Sets default flags for title field.
        """
        if self._titleField == None:
            self._titleField = TextField()
            self._titleField.setDefaultWidth(self.getDistance('titleFieldWidth', 1, True))
            self._titleField.setDefaultFontSize(self.TITLEFIELD_FONTSIZE)
            self._titleField.setAutotruncate(False)
            #self._titleField.setAutoscale(self._autosizeFlag, self._autosizeKeepAspectRatioFlag)
            self.titleField().setAutoscale(True, False)
            self._titleField.setPenColor(self.TITLE_COLOR)
    
    def titleIsSet(self):
        """ Returns True if test field has been set to a non empty string.
        """
        if self._titleField != None and not self._titleField.empty():
            return True
        return False    
    
    def setTitle(self, title):
        """ Sets title text.
        """
        self._initTitleField()
        self._titleField.setFont(self.font())
        self._titleField.setText(title)
        self.scheduleRearangeContent()
        self.update()
        
    def titleField(self):
        return self._titleField
    
    def title(self):
        if self.titleIsSet():
            return self._titleField.text()
        return None
        
    def _initTextField(self):
        if self._textField == None:
            self._textField = TextField()
            self._textField.setFontSizeRange(self.TEXTFIELD_FONTSIZE_MIN, self.TEXTFIELD_FONTSIZE_MAX)
            self._textField.setDefaultWidth(self.getDistance('textFieldWidth', 1, True))
            self._textField.setDefaultHeight(self.getDistance('textFieldHeight', 1, True))
            self._textField.setDefaultFontSize(self.TEXTFIELD_FONTSIZE)
            self._textField.setAutosizeFont(self.TEXTFIELD_AUTOSIZE_FONT_FLAG)
            self._textField.setAutotruncate(self.TEXTFIELD_AUTOTRUNCATE_TEXT_FLAG)
            self._textField.setOutputFlags(self.TEXTFIELD_FLAGS)
            self._textField.setAutoscale(self._autosizeFlag, self._autosizeKeepAspectRatioFlag)
        
    def textFieldIsSet(self):
        """ Returns True if text field text has been set to an non empty string.
        """
        if self._textField != None and not self._textField.empty():
            return True
        return False
    
    def setText(self, text):
        """ Sets text for text field.
        """
        #logging.debug(self.__class__.__name__ +": setText() - %s (%s)" % (str(text), str(type(text))))
        self._initTextField()
        self._textField.setFont(self.font())
        self._textField.setText(text)
        
        self.scheduleRearangeContent()
        
        self.setToolTip(self._textField.text())
        self.update()
        
    def textField(self):
        """ Returns TextField object belonging to text field.
        """
        return self._textField
    
    def text(self):
        """ Returns text of text field.
        """
        if self.textFieldIsSet():
            return self._textField.text()
        return None
        
    def setTextFieldAutosizeFont(self, auto):
        """ Sets auto resize flag of text field.
        """
        self._initTextField()
        self._textField.setAutosizeFont(auto)
        self.scheduleRearangeContent()
        
    def setTextFieldAutotruncateText(self, auto):
        """ Sets auto truncate flag of text field.
        """
        self._initTextField()
        self._textField.setAutotruncate(auto)
        self.scheduleRearangeContent()
    
    def setMaximumSize(self, *attr):
        QWidget.setMaximumSize(self, *attr)
        self.scheduleRearangeContent()
    
    def setMinimumSize(self, *attr):
        QWidget.setMinimumSize(self, *attr)
        self.scheduleRearangeContent()
    
    def sizeHint(self):
        """ Calculates needed space for widget content.
        """
        #if not self._autosizeFlag:
        #    return QSize(self.WIDTH, self.HEIGHT)
        
        self._scaleWidth = 1         # for getDistance()
        self._scaleHeight = 1
        
        neededWidth = self.getDistance('leftMargin', 1) + self.getDistance('rightMargin', 1)
        neededHeight = self.getDistance('topMargin', 1) + self.getDistance('bottomMargin', 1)
        
        titleFieldWidth = 0
        titleFieldHeight = 0
        titleIsSet = self.titleIsSet()
        if titleIsSet:
            titleFieldWidth = self.getDistance('titleFieldWidth', 1)
            titleFieldHeight += self.getDistance('titleFieldHeight', 1)
        
        textFieldWidth = 0
        textFieldHeight = 0
        if self.textFieldIsSet():
            textFieldWidth = self._textField.getWidth()
            textFieldHeight += self._textField.getHeight()
        bodyWidgetWidth = 0
        bodyWidgetHeight = 0
        if self._bodyWidget:
            if self._bodyWidget.parent() != self:
                self._bodyWidget = None
            else:
                sh = self._bodyWidget.sizeHint()
                bodyWidgetWidth = sh.width()
                bodyWidgetHeight = sh.height()
            
        imageSizeF = self.imageSizeF()
        bodyWidth = max(textFieldWidth, bodyWidgetWidth, imageSizeF.width())
        bodyHeight = max(textFieldHeight, bodyWidgetHeight, imageSizeF.height())
        
        if titleIsSet and bodyHeight != 0:
            # gap between title and text
            neededHeight += self.getDistance('bottomMargin', 1)
            
        neededWidth += max(bodyWidth, titleFieldWidth)
        neededHeight += titleFieldHeight + bodyHeight
        
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
    
    def resize(self, width, height):
        self.WIDTH = width / self.zoomFactor()
        self.HEIGHT = height / self.zoomFactor()
        ZoomableWidget.resize(self, self.width(), self.height())
    
    def autosize(self, skipSizeHint=False):
        """ Calculates scale factors and resizes widget accordingly.
        
        Calculates by which factor width and height of this widget should be scaled
        depending on the content of the widget.
        If skipSizeHint is True only the resize part is performed.
        """
        #logging.debug(self.__class__.__name__ +": autosize()")
        if not skipSizeHint:
            neededSpace = self.sizeHint()
            neededWidth = neededSpace.width()
            neededHeight = neededSpace.height()
        
            self._scaleWidth = 1.0 * neededWidth / self.WIDTH
            self._scaleHeight = 1.0 * neededHeight / self.HEIGHT
        
            if self._autosizeKeepAspectRatioFlag:
                self._scaleWidth = min(self._scaleWidth, self._scaleHeight)
                self._scaleHeight = self._scaleWidth
                
        if self._bodyWidget:
            self._bodyWidget.move(self.getDistance("contentStartX"), self.getDistance("contentStartY"))
        
        ZoomableWidget.resize(self, self.width(), self.height())
        self.update()
        # update() is sometimes required
        # if content has changed but size did not.
        # user can expect repainting after calling autosize()? TODO: think about this 

    def rearangeContent(self):
        """ Checks which components have to be regarded for size calculation.
        
        If you want to make sure this function is called next time a distance is requested use call scheduleRearangeContent().
        This function is stronger than autosize, as it also recalculates dimensions of children (e.g. text field).
        If content did not change autosize is probably the better approach.  
        """
        self._rearangeContentFlag = False
        # does not work with box decay tree (needs correct sizes before on screen)
        #if not self.isVisible():
        #    return
        
        #logging.debug(self.__class__.__name__ +": rearangeContent() ")
        
        if self.textFieldIsSet():
            self._textField.setDefaultWidth(self.getDistance('textFieldWidth', 1, True))
            self._textField.setDefaultHeight(self.getDistance('textFieldHeight', 1, True))
            self._textField.calculateDimensions()
        if self.titleIsSet():
            self._titleField.setDefaultWidth(self.getDistance('titleFieldWidth', 1, True))
            self._titleField.setDefaultHeight(self.getDistance('titleFieldHeight', 1, True))
            self._titleField.calculateDimensions()
            
        self._distancesHaveToBeRecalculatedFlag = True
        if self._autosizeFlag:
            self.autosize()
    
    def scheduleCalculateDistances(self):
        """ Sets distancesHaveToBeRecalculatedFlag to True.
        
        Next time defineDistances() is called distances are recalculated even if zoom has not changed.
        """
        self._distancesHaveToBeRecalculatedFlag = True
        
    def noRearangeContent(self,no=True):
        """ Flag disables any rearanging.
        """
        self._noRearangeContentFlag=no
        
    def scheduleRearangeContent(self):
        """ Makes sure rearangeContent() will be called next time a distance is requested.
        
        See rearangeContent().
        """
        self._rearangeContentFlag = True
        
    def showEvent(self, event):
        """ Calls rearangeContent() if needed.
        """
        # hasattr for some reason important
        # sometimes a show event seems to occur before the constructor is called
#        if hasattr(self, "_rearangeContentFlag") and hasattr(self, "_noRearangeContentFlag") and \
#            self._rearangeContentFlag and not self._noRearangeContentFlag:

        if self._rearangeContentFlag and not self._noRearangeContentFlag:
            self.rearangeContent()
        ZoomableWidget.showEvent(self, event)
    
    def distances(self):
        """ Returns dictionary containing distances as defined in defineDistances().
        """
        return self._distances

    def defineDistances(self, keepDefaultRatio=False):
        """ Defines supported distances.
        
        The distances are needed for drawing widget content and are scaled according to current scale / zoom factors.
        The distances are stored in an dictionary (see distances()).
        """
        if self._rearangeContentFlag and not self._noRearangeContentFlag:# and self.isVisible():
            self.rearangeContent()
            
        scale = 1.0     # remove if works without
        if keepDefaultRatio:
            scaleWidth = 1.0
            scaleHeight = 1.0
        else:
            scaleWidth = self._scaleWidth
            scaleHeight = self._scaleHeight
            
        if not self._distancesHaveToBeRecalculatedFlag and \
            scaleWidth == self._distancesLastScaleWidth and \
            scaleHeight == self._distancesLastScaleHeight:
            return False

        #logging.debug(self.__class__.__name__ +": defineDistances() - scale = '"+ str(scale) +"'")
        
        self._distancesLastScale = scale
        self._distancesLastScaleWidth = scaleWidth
        self._distancesLastScaleHeight = scaleHeight
        self._distancesHaveToBeRecalculatedFlag = False
            
        self._distances = dict()
        self._distances['width'] = self.WIDTH * scale * scaleWidth
        self._distances['height'] = self.HEIGHT * scale * scaleHeight
        
        self._distances['frameTop'] = 1
        self._distances['frameLeft'] = 1
        self._distances['frameBottom'] = self._distances['height'] - 1
        self._distances['frameRight'] = self._distances['width'] - 1
        
        if self._arrowShape == self.ARROW_SHAPE_TOP:
            self._distances['frameTop'] = self.ARROW_SIZE * scale
            self._distances['height'] += self._distances['frameTop']
            self._distances['frameBottom'] = self._distances['height'] -1
        elif self._arrowShape == self.ARROW_SHAPE_RIGHT:
            self._distances['frameRight'] = self._distances['width']
            self._distances['width'] += self.ARROW_SIZE * scale
        elif self._arrowShape == self.ARROW_SHAPE_BOTTOM:
            self._distances['frameBottom'] = self._distances['height']
            self._distances['height'] += self.ARROW_SIZE * scale
        elif self._arrowShape == self.ARROW_SHAPE_LEFT:
            self._distances['frameLeft'] = self.ARROW_SIZE * scale
            self._distances['width'] += self._distances['frameLeft']
            self._distances['frameRight'] = self._distances['width'] -1
        
        self._distances['topMargin'] = self.TOP_MARGIN * scale
        self._distances['leftMargin'] = self.LEFT_MARGIN * scale
        self._distances['bottomMargin'] = self.BOTTOM_MARGIN * scale
        self._distances['rightMargin'] = self.RIGHT_MARGIN * scale
        
        self._distances['horizontalInnerMargin'] = self.HORIZONTAL_INNER_MARGIN * scale
        self._distances['verticalInnerMargin'] = self.VERTICAL_INNTER_MARGIN * scale
        
        self._distances['titleFieldX'] = self._distances['frameLeft'] + self._distances['leftMargin']
        self._distances['titleFieldY'] = self._distances['frameTop'] + self._distances['topMargin']
        if self.titleIsSet():
            self._distances['titleFieldWidth'] = self._titleField.getWidth() * scale
            self._distances['titleFieldHeight'] = self._titleField.getHeight() * scale
            self._distances['titleFieldBottom'] = self._distances['titleFieldY'] + self._distances['titleFieldHeight']
        else:
            self._distances['titleFieldHeight'] = 0
            self._distances['titleFieldBottom'] = self._distances['frameTop']
            self._distances['titleFieldWidth'] = self._distances['width'] - self._distances['leftMargin'] - self._distances['rightMargin']
        
        self._distances['contentStartX'] = self._distances['frameLeft'] + self._distances['leftMargin']
        self._distances['contentStartY'] = self._distances['titleFieldBottom'] + self._distances['topMargin']
        
        self._distances['textFieldX'] = self._distances['contentStartX']
        self._distances['textFieldY'] = self._distances['contentStartY']
        if self.textFieldIsSet():
            self._distances['textFieldWidth'] = self._textField.getWidth() * scale
            self._distances['textFieldHeight'] = self._textField.getHeight() * scale
        else:
            self._distances['textFieldWidth'] = self._distances['width'] - self._distances['textFieldX'] - self._distances['rightMargin']
            self._distances['textFieldHeight'] = self._distances['height'] - self._distances['textFieldY'] - self._distances['bottomMargin']
        self._distances['textFieldRight'] = self._distances['textFieldX'] + self._distances['textFieldWidth']
        
        return True     # indicates changes for overridden function of sub classes
        
    def getDistance(self, name, scale=None, keepDefaultRatio=False):
        """ Gets the length of the element called 'name'.
        """
        self.defineDistances(keepDefaultRatio)
        if scale == None:
            scale = self._scale
        elif scale == 1:
            scale = 1.0
        
        if name in self._distances:
            #logging.debug(self.__class__.__name__ +": getdistance() - name = '"+ name +"' - "+ str(self._distances[name]))
            return self._distances[name] * scale
        else:
            logging.warning(self.__class__.__name__ +": getdistance() - Unknown distance '"+ name +"'")
            return 0

    def width(self):
        """ Returns width of this widget.
        """
        #if self.parent() and self.parent().layout():
        #    return QWidget.width(self)
        return self.getDistance('width')
    
    def height(self):
        """ Returns height of this widget.
        """
        # TODO: implement this more flexible regarding different QSizePolicies (also width())
        #if self.parent() and self.parent().layout():
        #    return QWidget.height(self)
        return self.getDistance('height')
    
    def isTitlePoint(self, point):
        """ Returns True if this point is part of the tile field, otherwise False is returned.
        """
        if not self.titleIsSet():
            return False
        if point.y() >= self.getDistance("titleFieldY") and point.y() <= self.getDistance("titleFieldBottom"):
            return True
        return False
    
    def defineRectBackgroundShape(self, painter):
        """ Draws background for rectangular shape.
        """
        l = self.getDistance('frameLeft')
        t = self.getDistance('frameTop')
        r = self.getDistance('frameRight')
        b = self.getDistance('frameBottom')
        myRect = QRectF(l, t, r - l , b - t)
        self._backgroundShapePath = QPainterPath()
        self._backgroundShapePath.addRect(myRect)
    
    def defineCircleBackgroundShape(self, painter):
        """ Draws background for circular shape.
        """
        w = self.width()
        h = self.height()
        r = min(w, h) - 3   # radius
        
        self._backgroundShapePath = QPainterPath()
        self._backgroundShapePath.addEllipse(0.5 * (w -r), 0.5 * (h -r), r, r)
    
    def defineRoundRectBackgroundShape(self, painter):
        """ Draws background for rectangular shape with rounded corners.
        """ 
        r = (self.ROUNDRECT_RADIUS) * self._scale
        w = self.width()# - 2
        h = self.height()# - 2
        
        w = self.getDistance("frameRight")
        h = self.getDistance("frameBottom")
        t = self.getDistance("frameTop")
        l = self.getDistance("frameLeft")
        
        # Prevent nasty lines when box too small
        f = 0.8  
        r = min(r, f * h, f * w)
        
        self._backgroundShapePath = QPainterPath()
        self._backgroundShapePath.moveTo(w, r + t)
        self._backgroundShapePath.arcTo(w - r, t, r, r, 0, 90)
        self._backgroundShapePath.lineTo(r + l, t)
        self._backgroundShapePath.arcTo(l, t, r, r, 90, 90)
        self._backgroundShapePath.lineTo(l, h - r)
        self._backgroundShapePath.arcTo(l, h - r, r, r, 180, 90)
        self._backgroundShapePath.lineTo(w - r, h)
        self._backgroundShapePath.arcTo(w - r, h - r, r, r, 270, 90)
        self._backgroundShapePath.closeSubpath()
        self._backgroundShapePath = self._backgroundShapePath.simplified()
        
    def defineArrowBackgroundShape(self):
        if not hasattr(self._backgroundShapePath, "united"):
            logging.warning(self.__class__.__name__ +": defineArrowBackgroundShape() - Upgrade your Qt version at least to 4.3 to use this feature. Aborting...")
            return
        
        #logging.debug(self.__class__.__name__ +":defineArrowBackgroundShape()")
        
        offset = 0
        if self._backgroundShape == "ROUNDRECT":
            offset = (self.ROUNDRECT_RADIUS) * self._scale * 0.2
        p = self._backgroundShapePath.toFillPolygon()
        #print "background shape", [p[i] for i in range(len(p))]
        arrowPath = None
        if self._arrowShape == self.ARROW_SHAPE_TOP:
            arrowPath = QPainterPath()
            arrowPath.moveTo(self.getDistance('frameLeft'), self.getDistance('frameTop') + offset) 
            arrowPath.lineTo(0.5 * self.width(), 1)
            arrowPath.lineTo(self.getDistance('frameRight'), self.getDistance('frameTop') + offset)
            arrowPath.closeSubpath()
        elif self._arrowShape == self.ARROW_SHAPE_RIGHT:
            arrowPath = QPainterPath()
            arrowPath.moveTo(self.getDistance('frameRight') - offset, self.getDistance('frameTop')) 
            arrowPath.lineTo(self.width(), 0.5 * self.height())
            arrowPath.lineTo(self.getDistance('frameRight') - offset, self.getDistance('frameBottom'))
            arrowPath.closeSubpath()
        elif self._arrowShape == self.ARROW_SHAPE_BOTTOM:
            arrowPath = QPainterPath()
            arrowPath.moveTo(self.getDistance('frameLeft'), self.getDistance('frameBottom') - offset) 
            arrowPath.lineTo(0.5 * self.width(), self.height())
            arrowPath.lineTo(self.getDistance('frameRight'), self.getDistance('frameBottom') - offset)
            arrowPath.closeSubpath()
        elif self._arrowShape == self.ARROW_SHAPE_LEFT:
            arrowPath = QPainterPath()
            arrowPath.moveTo(self.getDistance('frameLeft') + offset, self.getDistance('frameTop')) 
            arrowPath.lineTo(1, 0.5 * self.height())
            arrowPath.lineTo(self.getDistance('frameLeft') + offset, self.getDistance('frameBottom'))
            arrowPath.closeSubpath()

        if arrowPath:
            self._backgroundShapePath = arrowPath.united(self._backgroundShapePath).simplified()
    
    def drawHeaderBackground(self, painter):
        """ Color background of title in frame pen color.
        """
        if not self._colorHeaderBackgroundFlag or self._backgroundShapePath == None:
            # Cannot color background
            return
        
        if hasattr(self._backgroundShapePath, "intersected"):
            # Available since Qt 4.3
            topRectPath = QPainterPath()
            topRectPath.addRect(QRectF(0, 0, self.getDistance('width'), self.getDistance('titleFieldBottom')))
            headerPath = topRectPath.intersected(self._backgroundShapePath)
            painter.setPen(QColor(self.framePenColor))
            painter.setBrush(self.framePenColor)
            painter.drawPath(headerPath)
            return
        
        # Fallback for Qt versions prior to 4.3 
        
        backgroundShapePolygon = self._backgroundShapePath.toFillPolygon()
        headerPolygonPoints = []
        i = 0
        headerBottom = 0
        nearlyBottom = 0
        if self.isSelected():
            selectedWidth =  (self.SELECTED_FRAME_WIDTH +4) * self.scale()
        else:
            selectedWidth = 0
        while i < backgroundShapePolygon.count():
            thisP = backgroundShapePolygon.value(i)
            i += 1
            # selectedWidth prevents horizontal line in SELECT_COLOR
            if thisP.y() <= self.getDistance('titleFieldBottom') - selectedWidth:
                if thisP.y() > headerBottom:
                    headerBottom = thisP.y()
                elif thisP.y() > nearlyBottom:
                    nearlyBottom = thisP.y()
                headerPolygonPoints.append(thisP)
        headerPolygon = QPolygonF(headerPolygonPoints)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.framePenColor)
        titleBgPath = QPainterPath()
        titleBgPath.addPolygon(headerPolygon)
        painter.drawPath(titleBgPath)
        
        headerBottom = nearlyBottom     # test whether second lowest header point works better
        if (self._backgroundShape == 'ROUNDRECT' or self._backgroundShape == 'RECT') and headerBottom < self.getDistance('titleFieldBottom'):
        #if (headerBottom) < self.getDistance('titleFieldBottom'):        # This condition unfortunately does not work correctly on round shapes.
            # backgroundShapePolygon does not have a sufficient number of points at the straight lines on the side, so this is a work around.
            # This is not a clean solution as most functions should be independent from chosen backgroundShape.
            xBorder = 0
            if self.isSelected():
                # do not paint on frame
                if headerBottom == 0:
                    headerBottom = self.SELECTED_FRAME_WIDTH - 2
                xBorder = self.SELECTED_FRAME_WIDTH - 2
            #painter.setPen(Qt.NoPen)
            painter.setPen(self.framePenColor)
            painter.drawRect(QRectF(xBorder, headerBottom, self.width() - 2 * xBorder - 2, self.getDistance('titleFieldBottom') - headerBottom))
        
    def drawTitle(self, painter):
        """ Tells TextField object of title to draw title on widget.
        """
        if not self.titleIsSet():
            return
        self.drawHeaderBackground(painter)
        
        painter.setPen(QPen())
        self._titleField.paint(painter, self.getDistance('titleFieldX'), self.getDistance('titleFieldY'), self._scale)
        
    def drawTextField(self, painter):
        """ Tells TextField object of text field to draw title on widget.
        """
        if not self.textFieldIsSet():
                return

        painter.setPen(QPen())
        self._textField.paint(painter, self.getDistance('textFieldX'), self.getDistance('textFieldY'), self._scale)
        
    def drawBody(self, painter):
        """ This function calls drawTextField() and drawImage().
        
        Inheriting classes should overwrite this function if they wish to draw different things or alter the order of drawing.
        """
        self.drawTextField(painter)
        self.drawImage(painter)
    
    def contentRect(self):
        frame_width = 2
        if self.isSelected():
            frame_width = self.SELECTED_FRAME_WIDTH
        return QRect(frame_width, self.getDistance("titleFieldBottom"),
                     self.width() - 2* frame_width -1,
                     self.height() - self.getDistance("titleFieldBottom") - frame_width -1)
        
    def paintEvent(self, event):
        """ Reacts on paint event and calls paint() method.
        """
        #logging.debug(self.__class__.__name__ +": paintEvent()")
        painter = QPainter(self)
        if isinstance(self.parent(), VispaWidget):
            painter.setClipRegion(event.region().intersected(QRegion(self.parent().contentRect().translated(- self.pos()))))
        else:
            painter.setClipRegion(event.region())

        if self.zoom() > 30:
            painter.setRenderHint(QPainter.Antialiasing)
        else:
            painter.setRenderHint(QPainter.Antialiasing, False)

        self.paint(painter)
        ZoomableWidget.paintEvent(self, event)
        
    def paint(self, painter, event=None):
        """ Takes care of painting widget content on given painter.
        """
        if not self._backgroundGradientEnabledFlag or painter.redirected(painter.device()):
            # TODO: find condition which fits QPixmap but not QPicture (pdf export)
            # e.q. QPixmap.grabWidget()
            backgroundBrush = self.fillColor1
        else:
            backgroundBrush = QLinearGradient(0, self.getDistance('titleFieldBottom'), 0, self.height())
            backgroundBrush.setColorAt(0, self.fillColor1)
            backgroundBrush.setColorAt(1, self.fillColor2)
        
        painter.setPen(self.framePenColor)
        painter.pen().setJoinStyle(Qt.RoundJoin)
        painter.setBrush(backgroundBrush)
        
        if self._backgroundShape == 'CIRCLE':
            self.defineCircleBackgroundShape(painter)
        elif self._backgroundShape == 'ROUNDRECT':
            self.defineRoundRectBackgroundShape(painter)
        else:
            self.defineRectBackgroundShape(painter)
        
        if self._arrowShape != None:
            self.defineArrowBackgroundShape()
            
        painter.drawPath(self._backgroundShapePath)
        
        self.drawTitle(painter)
        self.drawBody(painter)
        
        if self.isSelected():
            # color frame
            framePen = QPen(self.SELECT_COLOR)
            framePen.setWidth(self.SELECTED_FRAME_WIDTH)
            painter.setPen(framePen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self._backgroundShapePath)
        
    def setDragReferencePoint(self, pos):
        self._dragMouseXrel = pos.x()
        self._dragMouseYrel = pos.y()
        
    def dragReferencePoint(self):
        return QPoint(self._dragMouseXrel, self._dragMouseYrel)
        
    def resetMouseDragOffset(self):
        self._dragMouseXrel = 0
        self._dragMouseYrel = 0
        
    def mousePressEvent(self, event):
        """ Register mouse offset for dragging and calls select().
        """
        parentIsVispaWidgetOwner = isinstance(self.parent(), VispaWidgetOwner)
        if event.modifiers() == Qt.ControlModifier:
            # allow deselect of individual widgets in selection
            self.select(not self.isSelected(), True)
        elif parentIsVispaWidgetOwner and not self.isSelected():
            self.select(True)
            
        if not self._dragableFlag:
            return
        self._dragMouseXrel = event.x()
        self._dragMouseYrel = event.y()
        
        if parentIsVispaWidgetOwner:
            self.parent().initWidgetMovement(self)

    def mouseMoveEvent(self, event):
        """ Call dragWidget().
        """
        #logging.debug("%s: mouseMoveEvent()" % self.__class__.__name__)
        if bool(event.buttons() & Qt.LeftButton):
            self.dragWidget(self.mapToParent(event.pos()))
        return
    
    def mouseReleaseEvent(self, event):
        #logging.debug("%s: mouseReleaeEvent()" % self.__class__.__name__)
        if self._dragMouseXrel != 0 and self._dragMouseYrel != 0:
            self.resetMouseDragOffset()
            self.emit(SIGNAL("dragFinished"))
        
    def keyPressEvent(self, event):
        """ Calls delete() method if backspace or delete key is pressed when widget has focus.
        """
        if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
            if hasattr(self.parent(), "multiSelectEnabled") and self.parent().multiSelectEnabled() and \
             hasattr(self.parent(), "selectedWidgets") and len(self.parent().selectedWidgets()) > 1:
                # let parent handle button event if multi-select is enabled
                self.parent().setFocus(Qt.OtherFocusReason)
                QCoreApplication.instance().sendEvent(self.parent(), event)
            else:
                self.emit(SIGNAL("deleteButtonPressed"))
                self.delete()
        else:
            # let parent handle all other events
            self.parent().setFocus(Qt.OtherFocusReason)
            QCoreApplication.instance().sendEvent(self.parent(), event)
            
    def delete(self):
        """ Deletes this widget.
        """
        if not self.isDeletable():
            logging.warning(self.__class__.__name__ +": delete() - Tried to remove undeletable widget. Aborting...")
            return False
        
        if isinstance(self.parent(), VispaWidgetOwner):
            self.parent().widgetAboutToDelete(self)

        self.deleteLater()
        self.emit(SIGNAL("widgetDeleted"))
        return True

    def setPreviousDragPosition(self, position):
        self._previusDragPosition = position

    def previousDragPosition(self):
        """ Returns position from before previous drag operation.
        
        E.g. used for undo function.
        """
        #print "VispaWidget.previousDragPosition()", self._previusDragPosition
        return self._previusDragPosition
    
    def dragWidget(self, pPos):
        """ Perform dragging when user moves cursor while left mouse button is pressed.
        """
        if not self._dragableFlag:
            return
        
        self._previusDragPosition = self.pos()
        #pPos = self.mapToParent(event.pos())
        self.move(max(0,pPos.x() - self._dragMouseXrel), max(0,pPos.y() - self._dragMouseYrel))
        
        # Tell parent, a widget moved to trigger file modification.
        if isinstance(self.parent(), VispaWidgetOwner):
            self.parent().widgetDragged(self)

    def move(self, *target):
        """ Move widgt to new position.
        
        You can either give x and y coordinates as parameters or a QPosition object.
        """
        if len(target) == 1:
            # Got point as argument
            targetX = target[0].x()
            targetY = target[0].y()
        else:
            # Got x and y as arguments
            targetX = target[0]
            targetY = target[1]
        
        self._unzoomedPositionX = 1.0 * targetX / self.scale()
        self._unzoomedPositionY = 1.0 * targetY / self.scale()
        # In self.setZoome() the widgets position can be set with these values
        
        QWidget.move(self, targetX, targetY)
        
        #import traceback
        #traceback.print_stack()
        
    def setImage(self, image):
        """ The given image will be shown in the centre of the widget.
        
        Currently supported image types are QPixmap and QSvgRenderer.
        """
        self._image = image
    
    def drawImage(self, painter):
        """ Draws image onto the widget's centre. See setImage().
        """
        if not self._image:
            return

        if isinstance(self._image, QSvgRenderer):
            #rect = self.imageRectF(self._image.defaultSize().width() * self.scale(), self._image.defaultSize().height() * self.scale())
            self._image.render(painter, self.imageRectF())
        elif isinstance(self._image, QPixmap):
            #rect = self.imageRectF(self._image.width() * self.scale(), self._image.height() * self.scale())
            painter.drawPixmap(self.imageRectF(), self._image, QRectF(self._image.rect()))
        
        # debug
        #painter.drawRect(self.imageRectF())
        
    def imageSizeF(self):
        """ Returns QSizeF object representing the unzoomed size of the image. See setImage().
        """
        if not self._image:
            return QSizeF(0, 0)
        if isinstance(self._image, QSvgRenderer):    
            return QSizeF(self._image.defaultSize())
        if isinstance(self._image, QPixmap):
            return QSizeF(self._image.size())
        logging.warning(self.__class__.__name__ +": imageSizeF() - Unknown image type.")
        return QSizeF(0, 0)
        
    def imageRectF(self, width=None, height=None):
        """ Returns draw area as QRectF for drawImage.
        """
        if not width or not height:
            size = self.imageSizeF() * self.scale()
            width = size.width()
            height = size.height()
        
        if width > self.width() or height > self.height():
            widthScale = 1.0 * self.width() / width
            heightScale = 1.0 *self.height() /  height
            scale = min(widthScale, heightScale)
            width *= scale
            height *= scale
        
        rect =  QRectF((self.width() - width) * 0.5, (self.height() - height + self.getDistance("titleFieldBottom")) * 0.5, width, height)
        #print "rect ", rect
        return rect
    
    def boundingRect(self):
        return QRect(self.x(), self.y(), self.width(), self.height())
    
    def setBodyWidget(self, widget):
        """ Accepts any QWidget and displays into the body section.
        
        The body section is below the header.
        """
        self._bodyWidget = widget
        self._bodyWidget.setParent(self)
        if self.isVisible():
            self._bodyWidget.setVisible(True)
        self.scheduleRearangeContent()
        
    def bodyWidget(self):
        """ Returns body widget if there is one or None otherwise.
        """
        return self._bodyWidget
