from PyQt4.QtCore import QCoreApplication, QEvent, Qt
from PyQt4.QtGui import QMouseEvent,QWidget, QCursor

import logging

from Vispa.Gui.VispaWidgetOwner import VispaWidgetOwner
from Vispa.Gui.PortConnection import PortConnection
from Vispa.Gui.ConnectableWidget import ConnectableWidget

class ConnectableWidgetOwner(VispaWidgetOwner):
    """ Interface for classes containing ConnectableWidgets
    
    Only makes sense if implementing class also inherits QWidget or class inheriting QWidget.
    """
    
    def widgetMoved(self, widget):
        """ Updates connections attached to one of the widget's ports.
        """
        VispaWidgetOwner.widgetMoved(self, widget)
        if not isinstance(widget, ConnectableWidget):
            return
        
        self.updateAttachedConnection(widget)
        if self.multiSelectEnabled():
            for child in self.children():
                if child != widget and not isinstance(child, PortConnection) and hasattr(child, "isSelected") and child.isSelected():
                    self.updateAttachedConnection(child)
    
    def updateAttachedConnection(self, widget):
        # update attached connections
        for connection in [child for child in self.children() if isinstance(child, PortConnection)]:
            if connection.sourcePort() in widget.ports() or connection.sinkPort() in widget.ports():
                connection.updateConnection()

    def widgetAboutToDelete(self, widget):
        """ Calls deleteWidgetConnections().
        """
        self.deleteWidgetConnections(widget)
        VispaWidgetOwner.widgetAboutToDelete(self, widget)
  
    def getWidgetByName(self, name):
        """ Returns module with given name or None if there is no such one.
        """
        for widget in [child for child in self.children() if hasattr(child, 'title')]:
            if widget.title() == name:
                return widget
        return None
  
    def connectionAboutToDelete(self, connection):
        """ Passes information on to parent.
        """
        if isinstance(self.parent(), ConnectableWidgetOwner):
            self.parent().connectionAboutToDelete(connection)
    
    def updateConnections(self):
        """ Updates all connection.
        """
        #logging.debug(self.__class__.__name__ +": updateConnections()")
        for child in self.children():
            if isinstance(child, ConnectableWidgetOwner):
                child.updateConnections()
            if isinstance(child, PortConnection):
                child.updateConnection()
            
    def deleteSelectedConnections(self):
        """ Deletes all selected connections.
        """
        for connection in [child for child in self.children() if isinstance(child, PointToPointConnection)]:
            if connection.isSelected():
                connection.delete()
    
    def deleteWidgetConnections(self, widget):
        """ Deletes connection which are attached to one of the widget's ports. 
        """
        deletedConnection = False
        ports = widget.ports()
        for connection in [child for child in self.children() if isinstance(child, PortConnection)]:
            for port in ports:
                if connection.attachedToPort(port):
                    deletedConnection = True
                    connection.delete()
        return deletedConnection
    
    def deleteConnectionsAttachedToPort(self, port):
        """ Deletes connections which are attached to given PortWidget (source or sink port).
        """
        deletedConnection = False
        for connection in [child for child in self.children() if isinstance(child, PortConnection)]:
            if connection.attachedToPort(port):
                deletedConnection = True
                connection.delete()
                
        return deletedConnection
    
    def portConnection(self, port):
        """ Returns the PortConnection if there is a connection in this ConnectableWidgetOwner that is attached to the given port.
        
        Otherwise None will be returned.
        """
        for connection in [child for child in self.children() if isinstance(child, PortConnection)]:
            if connection.attachedToPort(port):
                return connection
        return None

    def propagateEventUnderConnectionWidget(self, connection, event):
        """ This function propagates an event to one of it's children.
        
        If a connection widget is clicked in an area where it does not draw the connection line, the event should be forwarded to the underlying widget if there is such one.
        However the default behavior of Qt is to propagate the event to the connection's parent. This should be an ConnectableWidgetOwner object.
        This function is a workaround searching for any child widget at event.pos() which is not the initial connection.
        If it finds such a widget a new event with correct position in the new widget's own frame is created and sent to the widget.
        This function calls grabMouse() on the found child. The child should make sure releaseMouse() will be called e.g. in mouseReleaseEvent().

        Currently supported events: QEvent.MouseButtonPress, QEvent.MouseButtonDblClick.
        """
        logging.debug("%s: propagateEventUnderConnectionWidget() - %s" % (self.__class__.__name__, str(event.type())))
        # Currently supported events: QEvent.MouseButtonDblClick, QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseMove.
        workspacePos = connection.mapToParent(event.pos())
        for child in reversed(self.children()):
            if not child==connection and isinstance(child,QWidget) and child.geometry().contains(workspacePos):
                # do not forward event to connections which do not cross the mouse click point, this is important to prevent infinite loop error
                if isinstance(child,PortConnection) and not child.belongsToRoute(workspacePos):
                    continue
#                if event.type() == QEvent.MouseButtonDblClick or \
#                    event.type() == QEvent.MouseButtonPress or \
#                    event.type() == QEvent.MouseButtonRelease or \
#                    event.type() == QEvent.MouseMove or \
#                    event.type() == QEvent.DragEnter or \
#                    event.type() == QEvent.Drop:

                childPos = child.mapFromParent(workspacePos)
                grandChild = child.childAt(childPos)
                if grandChild:
                    child = grandChild
                    childPos = child.mapFromParent(childPos)
                if event.type() == QEvent.MouseButtonPress:
                    child.grabMouse(QCursor(Qt.ClosedHandCursor))
                    child.setFocus()
#                    print "    MouseButtonPress", child
#                if event.type() == QEvent.MouseMove:
#                    print "    mouseMoveEvent", child
                newEvent = QMouseEvent(event.type(), childPos, event.button(), event.buttons(), event.modifiers())
                QCoreApplication.instance().sendEvent(child, newEvent)
                return True
        return False
