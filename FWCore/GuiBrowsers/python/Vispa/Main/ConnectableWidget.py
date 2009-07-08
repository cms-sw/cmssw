from PyQt4.QtCore import QCoreApplication, QRect
from PyQt4.QtGui import QMouseEvent
from PyQt4.QtSvg import QSvgRenderer

from Vispa.Main.VispaWidget import *
from Vispa.Main.PortWidget import *
from Vispa.Main.VispaWidgetOwner import *

class ConnectableWidget(VispaWidget, VispaWidgetOwner):
    """ Widget which can be connection by PortConnections to other selectable widgets.
    
    Supports showing source and sink ports.
    The widget is owner of PortWidgets.
    """
    
    BACKGROUND_SHAPE = 'ROUNDRECT'
    SHOW_PORT_NAMES = False
    SHOW_PORT_LINES = False
    
    # possible positions for port names
    PORT_NAMES_NEXT_TO_PORTS = 0
    PORT_NAMES_ABOVE_PORTS = 1
    
    # default position for port names
    PORT_NAMES_POSITION = PORT_NAMES_NEXT_TO_PORTS
    
    NO_VALID_PORT_NAMES_POSITION_MESSAGE = "No valid position for port names was set."
    
    PORT_LINES_TARGET_X = -1     # See setShowPortNames()
    PORT_LINES_TARGET_Y = -1
   
    def __init__(self, parent=None, name=None):
        """ Constructor.
        """
        self._ports = []
        self._showPortNames = False
        self._portNamesPosition = None
        self._showPortLines = False
        self._image = None
        VispaWidget.__init__(self, parent)
        self.setShowPortNames(self.SHOW_PORT_NAMES)
        self.setPortNamesPosition(self.PORT_NAMES_POSITION)
        self.setShowPortLines(self.SHOW_PORT_LINES)
        
        if name:
            self.setTitle(name)
    
    def setShowPortNames(self, show):
        """ If True the port name's will be drawn.
        
        The port names wont be on the port itself.
        Instead they will appear next to the port icons on the ConnectableWidget.
        """
        self._showPortNames = show
        
    def setPortNamesPosition(self, position):
        """ Sets position where port names will be shown.
        
        Possible values are self.PORT_NAMES_NEXT_TO_PORTS and self.PORT_NAMES_ABOVE_PORTS.
        """
        self._portNamesPosition = position
    
    def setShowPortLines(self, show):
        """ If True lines from all ports to a specific target point are drawn.
        
        The target point is defined by PORT_LINES_TARGET_X and PORT_LINES_TARGET_Y.
        If both of these values are -1 the target point is set to the widget's centre.
        """
        self._showPortLines = show
        
    def setImage(self, image):
        """ The given image will be shown in the centre of the widget.
        
        Currently supported imape types are QPixmap and QSvgRenderer.
        """
        self._image = image
    
    def getPortsHeight(self, portType):
        """ Returns height of all ports of given type.
        
        portType can either be 'sink" or 'source'.
        """
        if portType == "sink":
            ports = self.sinkPorts()
        elif portType == "source":
            ports = self.sourcePorts()
        else:
            return 0

        if len(ports) > 1:
            return ports[0].y() - ports[len(ports) -1].y() + 0.5 * self.getEffectivePortHeight(ports[len(ports) -1])
        elif len(ports) == 1:
            return self.getEffectivePortHeight(ports[0])
        else:
            return 0

    def sizeHint(self):
        """ Returns size needed to draw widget's content.
        """
        #logging.debug(self.__class__.__name__ + ": sizeHint()")
        # arrangePorts() needed because it will be called in rearnangeContent() after sizeHint()
        self.arrangePorts()
        
        neededWidth = self.getDistance('leftMargin', 1) + self.getDistance('rightMargin', 1)
        neededHeight = self.getDistance('topMargin', 1) + self.getDistance('bottomMargin', 1)
        
        # height        
        if self.titleIsSet():
            neededHeight += self.getDistance('titleFieldHeight', 1) + self.getDistance('bottomMargin', 1)

        sinkPortsHeight = self.getPortsHeight("sink") / self.scale()
        sourcePortsHeight = self.getPortsHeight("source") / self.scale()
        textFieldHeight = 0
        if self.textFieldIsSet():
            textFieldHeight += self.textField().getHeight()
        imageHeight = self.imageSizeF().height()
        neededHeight += max(sinkPortsHeight, sourcePortsHeight, textFieldHeight, imageHeight)
        
        # width
        titleWidth = 0
        if self.titleIsSet():
            titleWidth = self.getDistance('titleFieldWidth', 1)
            
        bodyWidth = 0
        sinkPortsWidth = 0
        sourcePortsWidth = 0
        if len(self.sinkPorts()) > 0:
            sinkPortsWidth = self.getDistance('leftMargin', 1) + PortWidget.WIDTH
        if len(self.sourcePorts()) > 0:
            sourcePortsWidth = self.getDistance('rightMargin', 1) + PortWidget.WIDTH
            
        if self._showPortNames:
            maxSinkTitleWidth = self._getMaxSinkTitleWidth()
            maxSourceTitleWidth = self._getMaxSourceTitleWidth()
            if self._portNamesPosition == self.PORT_NAMES_NEXT_TO_PORTS:
                bodyWidth += maxSinkTitleWidth + self.getDistance('rightMargin', 1) + maxSourceTitleWidth
            elif self._portNamesPosition == self.PORT_NAMES_ABOVE_PORTS:
                if maxSinkTitleWidth > PortWidget.WIDTH:
                    sinkPortsWidth = 0#self.getDistance('leftMargin', 1)
                if maxSourceTitleWidth > PortWidget.WIDTH:
                    sourcePortsWidth = 0#self.getDistance('rightMargin', 1)
                #bodyWidth += maxSinkTitleWidth + self.getDistance('rightMargin', 1) + maxSourceTitleWidth
                bodyWidth += maxSinkTitleWidth + maxSourceTitleWidth
            else:
                logging.waring(self.__class__.__name__ +": sizeHint() - "+ self.NO_VALID_PORT_NAMES_POSITION_MESSAGE)
        bodyWidth += sinkPortsWidth + sourcePortsWidth
                            
        if self.textFieldIsSet():
            bodyWidth += self.getDistance('textFieldWidth', 1)
            
        neededWidth += max(titleWidth, bodyWidth)
        
        return QSize(neededWidth, neededHeight)
    
    def defineDistances(self, keepDefaultRatio=False):
        """ Extends distances of VispaWidget by the additionally needed distances for displaying ports.
        """
        #if scale == None:
        #    scale = self.scale()
        scale = 1.0
        
        if not VispaWidget.defineDistances(self, keepDefaultRatio):
            return False
        if len(self.sinkPorts()) > 0:
            self.distances()['textFieldX'] += PortWidget.WIDTH * scale + self.distances()['leftMargin']
            self.distances()['textFieldRight'] = self.distances()['textFieldX'] + self.distances()['textFieldWidth']
            if self._showPortNames:
                self.distances()['textFieldX'] += self._getMaxSinkTitleWidth() + self.distances()['leftMargin']
                self.distances()['textFieldRight'] += self._getMaxSinkTitleWidth() + self.distances()['leftMargin']
                
        firstPortY = self.distances()['height'] - self.distances()['bottomMargin'] - PortWidget.HEIGHT * scale
        self.distances()['firstSinkX'] = self.distances()['leftMargin']
        self.distances()['firstSinkY'] = firstPortY
        
        if self.textFieldIsSet():
            self.distances()['firstSourceX'] = self.distances()['textFieldRight'] + self.distances()['leftMargin']
        #else:
        self.distances()['firstSourceX'] = self.distances()['width'] - self.distances()['leftMargin'] - PortWidget.WIDTH * scale
        self.distances()['firstSourceY'] = firstPortY
        
        return True
        
#    def scaleChanged(self):
#        """ Arranges ports when scale has changed.
#        """
#        VispaWidget.scaleChanged(self)
#        self.arrangePorts()

    def setZoom(self, zoom):
        """ Arranges ports when zoom has changed.
        """
        VispaWidget.setZoom(self, zoom)
        #self.arrangePorts()
        
    def mousePressEvent(self, event):
        """ Makes sure event is forwarded to both base classes.
        
        If position of event is within the dropArea of a port a QMouseEvent is sent to the port. See dropArea().
        """
        dropAreaPort = self.dropAreaPort(event.pos())
        if dropAreaPort and dropAreaPort.isDragable():
            dropAreaPort.grabMouse()
            newEvent = QMouseEvent(event.type(), dropAreaPort.mapFromParent(event.pos()), event.button(), event.buttons(), event.modifiers())
            QCoreApplication.instance().sendEvent(dropAreaPort, newEvent)
        else:
            VispaWidgetOwner.mousePressEvent(self, event)
            VispaWidget.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        """ Calls realeseMouse() to make sure the widget does not grab the mouse.
        
        Necessary because ConnectableWidgetOwner.propagateEventUnderConnectionWidget() may call grabMouse() on this widget.
        """
        logging.debug(self.__class__.__name__ +": mouseReleaseEvent()")
        self.releaseMouse()
        VispaWidget.mouseReleaseEvent(self, event)
         
    def delete(self):
        """ Deletes this widget.
        
        Asks parent (ConnectableWidgetOwner) to remove all connections attached to any of this widgets ports and deletes the widget.
        """
        if not self.isDeletable():
            return None
        if isinstance(self.parent(), VispaWidgetOwner):
            self.parent().widgetAboutToDelete(self)
        VispaWidget.delete(self)
            
    def ports(self):
        """ Returns list containing all source and sink port widgets.
        """
        return self._ports
    
    def addSinkPort(self, name, description=None):
        """ Adds sink port with name and optional description text.
        """
        port=SinkPort(self, name)
        self._addPort(port, description)
        return port
        
    def addSourcePort(self, name, description=None):
        """ Adds source port with name and optional description text.
        """
        port=SourcePort(self, name)
        self._addPort(port, description)
        return port
        
    def _addPort(self, port, description=None):
        self._ports.append(port)
        port.show()
        if description:
            self._ports[len(self._ports) - 1].setDescription(description)
        self.scheduleRearangeContent()
        
    def removePort(self, port):
        """ Removes given port if it is port of this widget.
        """
        if port in self._ports:
            self._ports.remove(port)
            port.setParent(None)
            port.deleteLater()
            
    def portExists(self, name, description=None):
        for port in self._ports:
            if port.name() == name and port.description() == description:
                return True
        return False
         
        
    def removePorts(self, filter=None):
        """ Remove registered ports.
        
        If filter is "sink" only sinks are removed, if it is "source" only sources are removed, otherwise all ports are removed.
        """
        if filter and (filter != "sink" and filter != "source"):
            filter = None
        ports = self._ports[:]
        for port in ports:
            if not filter or port.portType() == filter:
                self._ports.remove(port)
                port.setParent(None)
                port.deleteLater()
        self.scheduleRearangeContent()
        self.update()                    
        
    def sinkPorts(self):
        """ Returns list of all sink ports set.
        """
        return [port for port in self._ports if port.portType() == "sink"]
        def isSink(port):
            return port.portType() == 'sink'
        return filter(isSink, self._ports)
    
    def sourcePorts(self):
        """ Returns list of all source ports set.
        """
        return [port for port in self._ports if port.portType() == "source"]
        def isSource(port):
            return port.portType() == 'source'
        return filter(isSource, self._ports)
    
    def sinkPort(self, name):
        """ Returns sink port with given name or None if no such port is found.
        """
        return self.port(name, 'sink')
    
    def sourcePort(self, name):
        """ Returns source port with given name or None if no such port is found.
        """
        return self.port(name, 'source')
        
    def port(self, name, type):
        """ Returns port with given name and of given type.
        """
        for port in self._ports:
            if port.portType() == type and port.name() == name:
                return port
        return None
    
    def _getMaxPortTitleWidth(self, type):
        if type == 'sink':
            ports = self.sinkPorts()
        elif type == 'source':
            ports = self.sourcePorts()
        else:
            return 0
        
        if len(ports) < 1:
            return 0
        return max([port.titleField().getWidth() for port in ports])
    
    def _getMaxSinkTitleWidth(self):
        return self._getMaxPortTitleWidth('sink')
    
    def _getMaxSourceTitleWidth(self):
        return self._getMaxPortTitleWidth('source')
    
    def getEffectivePortHeight(self, port):
        """ Returns the bigger value of the source height and the height of the port name text field.
        """ 
        portHeight = port.height()
        if not self._showPortNames:
            return portHeight
        
        titleHeight = port.titleField().getHeight() * self.scale()
        
        if self._portNamesPosition == self.PORT_NAMES_NEXT_TO_PORTS:
            if portHeight > titleHeight:
                return portHeight
            else:
                return titleHeight
        elif self._portNamesPosition == self.PORT_NAMES_ABOVE_PORTS:
            return portHeight + titleHeight
        logging.waring(self.__class__.__name__ +": getEffectivePortHeight() - "+ self.NO_VALID_PORT_NAMES_POSITION_MESSAGE)
        return 0

    def rearangeContent(self):
        """ Arranges ports after content is rearranged by VispaWidget.
        """
        VispaWidget.rearangeContent(self)
        self.arrangePorts()     # has to be after rearangeContent(), prevents infinite loop (..getDistance())

    def arrangePorts(self, filter=None):
        """ Sets positions of set ports depending on zoom factor.
        
        If filter is set it may be 'sink' or 'source'.
        """
        if filter and (filter != "sink" and filter != "source"):
            filter = None
        sinkCounter = 0
        sourceCounter = 0
        
        sinkX = self.getDistance('firstSinkX')
        sinkY = self.getDistance('firstSinkY')
        sourceX = self.getDistance('firstSourceX')
        sourceY = self.getDistance('firstSourceY')
        
        for port in self._ports:
            if port.portType() == 'sink' and (not filter or filter == "sink"):
                sinkCounter += 1
                port.move(sinkX, sinkY)
                sinkY -= self.getDistance('topMargin') + self.getEffectivePortHeight(port)  #+ PortWidget.HEIGHT * self.scale()
                
            elif port.portType() == 'source' and (not filter or filter == "source"):
                sourceCounter += 1
                port.move(sourceX, sourceY)
                sourceY -= self.getDistance('topMargin') + self.getEffectivePortHeight(port) # + PortWidget.HEIGHT * self.scale()

    def paint(self, painter, event=None):
        """ Takes care of painting widget content on given painter.
        """
        VispaWidget.paint(self, painter, event)
        self.drawPortLines(painter)
        self.drawImage(painter)
        self.drawPortNames(painter)
    
    def drawPortNames(self, painter):
        """ Paints port names next to PortWidget.
        
        See setShowPortNames().
        """
        if not self._showPortNames:
            return
        
        # factor should be 0.5, but text height is calculated to big
        titleHeightFactor = 0.4
        
        if self._portNamesPosition == self.PORT_NAMES_NEXT_TO_PORTS:
            for port in self.sinkPorts():
                if port.titleField():
                    port.titleField().paint(painter, port.x() + self.getDistance('rightMargin') + port.width(), port.y() - titleHeightFactor * port.getDistance('titleFieldHeight'), self.scale())
                
            for port in self.sourcePorts():
                if port.titleField():
                    #logging.debug(self.__class__.__name__ +": drawPortNames() - "+ port.name() +", "+ str(port.titleField()._autoscaleFlag))
                    port.titleField().paint(painter, port.x() - port.getDistance('titleFieldWidth') - self.getDistance('rightMargin'), port.y() - titleHeightFactor * port.getDistance('titleFieldHeight'), self.scale())
        elif self._portNamesPosition == self.PORT_NAMES_ABOVE_PORTS:
            painter.pen().setWidth(2)
            for port in self.sinkPorts():
                if port.titleField():
                    port.titleField().paint(painter, self.getDistance('firstSinkX'), port.y() - titleHeightFactor * port.getDistance('titleFieldHeight') - port.height(), self.scale())
                
            for port in self.sourcePorts():
                if port.titleField():
                    #logging.debug(self.__class__.__name__ +": drawPortNames() - "+ port.name() +", "+ str(port.titleField()._autoscaleFlag))
                    port.titleField().paint(painter, self.width() - port.getDistance('titleFieldWidth')- port.width()*0.5, port.y() - titleHeightFactor * port.getDistance('titleFieldHeight') - port.height(), self.scale())
        else:
            logging.waring(self.__class__.__name__ +": drawPortNames() - "+ self.NO_VALID_PORT_NAMES_POSITION_MESSAGE)


    def drawPortLines(self, painter):
        """ Draws lines from every port to a common point.
        
        See setShowPortLines().
        """
        if not self._showPortLines:
            return
        
        if self.PORT_LINES_TARGET_X == -1 and self.PORT_LINES_TARGET_Y == -1:
            targetPoint = QPoint(self.width(), self.height() + self.getDistance("titleFieldBottom")) * 0.5
        else:
            targetPoint = QPoint(self.PORT_LINES_TARGET_X, self.PORT_LINES_TARGET_Y) * self.scale()
        
        painter.setPen(QPen(QColor('black')))
        painter.pen().setWidth(1)
        
        for port in self.ports():
            painter.drawLine(port.connectionPoint("widget"), targetPoint)
    
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
    
    def dropArea(self, port):
        """ A drop area is a QRect in which the ConnectableWidget accepts dropping of PortWidgets to create connections.
        
        The area is greater than the port itself to make dropping easier.
        """
        if self._showPortNames:
            return port.frameGeometry().united(port.titleField().getDrawRect())
        topMargin = self.getDistance("topMargin")
        topMarginHalf = 0.5 * topMargin
        frameGeometry = port.frameGeometry()
        return QRect(frameGeometry.x() - topMarginHalf,
                     frameGeometry.y() - topMarginHalf,
                     frameGeometry.width() + topMargin,
                     frameGeometry.height() + topMargin)
            
    def dropAreaPort(self, position):
        """ If a port's drop area is associated with position the port is returned.
        
        If there is no drop area associated with the position None is returned.
        See dropArea().
        """
        for port in self._ports:
            if self.dropArea(port).contains(position):
                return port
        return None
        
