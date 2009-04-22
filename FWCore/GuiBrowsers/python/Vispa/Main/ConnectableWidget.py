from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Vispa.Main.VispaWidget import *
from Vispa.Main.PortWidget import *
from Vispa.Main.VispaWidgetOwner import *
from Vispa.Main.ConnectableWidgetOwner import *

class ConnectableWidget(VispaWidget, VispaWidgetOwner):
    """ Widget which can be connection by PortConnections to other selectable widgets.
    
    Supports showing source and sink ports.
    The widget is owner of PortWidgets.
    """
    
    BACKGROUND_SHAPE = 'ROUNDRECT'
    SHOW_PORT_NAMES = False
   
    def __init__(self, parent=None, name=None):
        """ Constructor.
        """
        self._ports = []
        self._showPortNames = False
        VispaWidget.__init__(self, parent)
        self.setShowPortNames(self.SHOW_PORT_NAMES)
        
        if name:
            self.setName(name)
    
    def setShowPortNames(self, show):
        """ If True the port name's will be drawn.
        
        The port names wont be on the port itself.
        Instead they will appear next to the port icons on the ConnectableWidget.
        """
        self._showPortNames = show
    
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
        neededHeight += max(sinkPortsHeight, sourcePortsHeight, textFieldHeight)
        
        # width
        titleWidth = 0
        if self.titleIsSet():
            titleWidth = self.getDistance('titleFieldWidth', 1)
            
        bodyWidth = 0
        if len(self.sinkPorts()) > 0:
            bodyWidth += self.getDistance('leftMargin', 1) + PortWidget.WIDTH
        if len(self.sourcePorts()) > 0:
            bodyWidth += self.getDistance('rightMargin', 1) + PortWidget.WIDTH
            
        if self._showPortNames:
            maxSinkTitleWidth = self._getMaxSinkTitleWidth()
            maxSourceTitleWidth = self._getMaxSourceTitleWidth()
            bodyWidth += maxSinkTitleWidth + self.getDistance('rightMargin', 1) + maxSourceTitleWidth
                            
        if self.textFieldIsSet():
            bodyWidth += self.getDistance('textFieldWidth', 1)
            
        neededWidth += max(titleWidth, bodyWidth)
        
        return QSize(neededWidth, neededHeight)
    
    def defineDistances(self, scale=None, keepDefaultRatio=False):
        """ Extends distances of VispaWidget by the additionally needed distances for displaying ports.
        """
        if scale == None:
            scale = self.scale()
        
        if not VispaWidget.defineDistances(self, scale, keepDefaultRatio):
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
        self.arrangePorts()
        
    def mousePressEvent(self, event):
        """ Makes sure event is forwarded to both base classes.
        
        If position of event is within the dropArea of a port a QMouseEvent is sent to the port. See dropArea().
        """
        dropAreaPort = self.dropAreaPort(event.pos())
        if dropAreaPort and dropAreaPort.isDragable():
            dropAreaPort.grabMouse()
            newEvent = QMouseEvent(event.type(), dropAreaPort.mapFromParent(event.pos()), event.button(), event.buttons(), event.modifiers())
            qApp.sendEvent(dropAreaPort, newEvent)
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
        if isinstance(self.parent(), ConnectableWidgetOwner):
            self.parent().widgetAboutToDelete(self)
        VispaWidget.delete(self)
        
    def setName(self, name):
        """ Sets name of this widget.
        """
        self.setTitle(name)
        
    def name(self):
        """ Returns name of this widget or an empty string if none has been set.
        """
        if self.titleIsSet():
            return self.title()
        return ''
    
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
        if description:
            self._ports[len(self._ports) - 1].setDescription(description)
        self.scheduleRearangeContent()  
        
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
        
        width = 0
        for port in ports:
            thisWidth = port.titleField().getWidth()
            if thisWidth > width:
                width = thisWidth
        return width
    
    def _getMaxSinkTitleWidth(self):
        return self._getMaxPortTitleWidth('sink')
    
    def _getMaxSourceTitleWidth(self):
        return self._getMaxPortTitleWidth('source')
    
    def getEffectivePortHeight(self, port):
        """ Returns the bigger value of the source height and the height of the port name text field.
        """ 
        portHeight = port.getHeight()
        if not self._showPortNames:
            return portHeight
        
        titleHeight = port.titleField().getHeight() * self.scale()
        
        if portHeight > titleHeight:
            return portHeight
        else:
            return titleHeight

    def rearangeContent(self):
        """ Arranges ports after content is rearranged by VispaWidget.
        """
        VispaWidget.rearangeContent(self)
        # has to be after rearangeContent(), prevents infinite loop (..getDistance())
        self.arrangePorts()
        #VispaWidget.autoresize(self)
        #self.arrangePorts()

    def arrangePorts(self):
        """ Sets positions of set ports depending on zoom factor.
        """
        #logging.debug(__name__ +": arragnePorts()")
        sinkCounter = 0
        sourceCounter = 0
        
        sinkX = self.getDistance('firstSinkX')
        sinkY = self.getDistance('firstSinkY')
        sourceX = self.getDistance('firstSourceX')
        sourceY = self.getDistance('firstSourceY')
        
        for port in self._ports:
            if port.portType() == 'sink':
                sinkCounter += 1
                port.move(sinkX, sinkY)
                sinkY -= self.getDistance('topMargin') + self.getEffectivePortHeight(port)  #+ PortWidget.HEIGHT * self.scale()
                
            elif port.portType() == 'source':
                sourceCounter += 1
                port.move(sourceX, sourceY)
                sourceY -= self.getDistance('topMargin') + self.getEffectivePortHeight(port) # + PortWidget.HEIGHT * self.scale()

    def paint(self, painter, event=None):
        """ Takes care of painting widget content on given painter.
        """
        VispaWidget.paint(self, painter, event)
        self.drawPortNames(painter)
    
    def drawPortNames(self, painter):
        """ Paints port names next to PortWidget.
        
        See setShowPortNames().
        """
        if not self._showPortNames:
            return
        
        # factor should be 0.5, but text height is calculated to big
        titleHeightFactor = 0.4
        
        for port in self.sinkPorts():
            if port.titleField():
                port.titleField().paint(painter, port.x() + self.getDistance('rightMargin') + port.getWidth(), port.y() - titleHeightFactor * port.getDistance('titleFieldHeight'), self.scale())
                
        for port in self.sourcePorts():
            if port.titleField():
                #logging.debug(self.__class__.__name__ +": drawPortNames() - "+ port.name() +", "+ str(port.titleField()._autoscaleFlag))
                port.titleField().paint(painter, port.x() - port.getDistance('titleFieldWidth') - self.getDistance('rightMargin'), port.y() - titleHeightFactor * port.getDistance('titleFieldHeight'), self.scale())
    
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
        
