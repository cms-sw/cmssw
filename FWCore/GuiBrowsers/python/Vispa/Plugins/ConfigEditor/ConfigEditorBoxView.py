import logging

from PyQt4.QtCore import Qt,QCoreApplication,SIGNAL

from Vispa.Main.Application import Application
from Vispa.Views.BoxDecayView import BoxDecayView
from Vispa.Gui.PortConnection import PortConnection,PointToPointConnection
from Vispa.Gui.ConnectableWidget import ConnectableWidget
from Vispa.Gui.WidgetContainer import WidgetContainer

class ConfigEditorBoxView(BoxDecayView):
    """
    """
    
    LABEL="ConfigEditor BoxView"
    
    def __init__(self, parent=None,label=""):
        logging.debug(__name__ + ": __init__")
        BoxDecayView.__init__(self, parent)
        self._connections = {}
        self._colors = [Qt.red, Qt.green, Qt.blue, Qt.cyan, Qt.magenta]
        self._colorIndex = 0
        PointToPointConnection.CONNECTION_THICKNESS=2
        self.setSortBeforeArranging(False)
        self._highlightedObjects=[]
        self._highlightedWidgets=[]

    def connections(self):
        return self._connections

    def setConnections(self, connections):
        """ Sets the connections between the objects.
        
        You need to call updateContent() in order to make the changes visible.
        """
        self._colorIndex = 0
        self._connections = connections
    
    def createBox(self, widgetParent, container, title, text):
        widget=BoxDecayView.createBox(self, widgetParent, container, title, text)
        widget.setToolTip("Double click to display this module with its dependencies")
        return widget
    
    def createConnections(self, operationId, widgetParent):
        for connection,values in self._connections.items():
            # Process application event loop in order to accept user input during time consuming drawing operation
            self._updateCounter+=1
            if self._updateCounter>=self.UPDATE_EVERY:
                self._updateCounter=0
                if not Application.NO_PROCESS_EVENTS:
                    QCoreApplication.instance().processEvents()
            # Abort drawing if operationId out of date
            if operationId != self._operationId:
                return None
            w1 = self.widgetByObject(connection[0])
            w2 = self.widgetByObject(connection[1])
            if w1 and w2:
                if hasattr(w1,"colorIndex"):
                    col = w1.colorIndex
                else:
                    self._colorIndex += 1
                    if self._colorIndex >= len(self._colors):
                        self._colorIndex = 0
                    col = self._colorIndex
                    w1.colorIndex=col
                    w2.colorIndex=col
                connectionWidget = self.createConnection(w1, values[0], w2, values[1], self._colors[col])
                connectionWidget.show()
        return True
    
    def highlight(self, objects=None):
        """ Mark objects.
        """
        if objects!=None:
            self._highlightedObjects=objects
        for widget in self._highlightedWidgets:
            if isinstance(widget,ConnectableWidget):
                widget.setColors(ConnectableWidget.PEN_COLOR,
                                 ConnectableWidget.FILL_COLOR1,
                                 ConnectableWidget.FILL_COLOR2)
            elif isinstance(widget,WidgetContainer):
                widget.setColors(WidgetContainer.PEN_COLOR,
                                 WidgetContainer.FILL_COLOR1,
                                 WidgetContainer.FILL_COLOR2)
            widget.update()
        self._highlightedWidgets=[]
        for object in self._highlightedObjects:
            widget = self.widgetByObject(object)
            if widget==None:
                continue
            if isinstance(widget,ConnectableWidget):
                widget.setColors(ConnectableWidget.PEN_COLOR.darker(),
                                 ConnectableWidget.FILL_COLOR1,
                                 ConnectableWidget.FILL_COLOR2)
            elif isinstance(widget,WidgetContainer):
                widget.setColors(WidgetContainer.PEN_COLOR.darker(),
                                 WidgetContainer.FILL_COLOR1,
                                 WidgetContainer.FILL_COLOR2)
            widget.update()
            self._highlightedWidgets+=[widget]

    def updateContent(self,overrideCheck=False):
        self._highlightedWidgets=[]
        ok=BoxDecayView.updateContent(self,overrideCheck)
        self.highlight()
        return ok

class ConnectionStructureView(ConfigEditorBoxView):
    LABEL="Connection structure"

class SequenceStructureView(ConfigEditorBoxView):
    LABEL="Sequence structure"

