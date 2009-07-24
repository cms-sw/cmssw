import logging

from PyQt4.QtCore import Qt

from Vispa.Main.BoxDecayTree import BoxDecayTree
from Vispa.Main.PortConnection import PortConnection

class ConfigBrowserBoxView(BoxDecayTree):
    """
    """
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        BoxDecayTree.__init__(self, parent)
        self._connections = []
        self._colors = [Qt.red, Qt.green, Qt.blue, Qt.cyan, Qt.magenta]
        self._colorIndex = 0

    def connections(self):
        return self._connections

    def setConnections(self, connections):
        """ Sets the connections between the objects.
        
        You need to call updateContent() in order to make the changes visible.
        """
        self._colorIndex = 0
        self._connections = connections
        
    def createConnections(self, operationId, widgetParent):
        for connection in self._connections:
            if operationId != self._operationId:
                break
            w1 = self._widgetByObject(connection[0])
            w2 = self._widgetByObject(connection[2])
            if w1 and w2:
                col = - 1
                if widgetParent:
                    children = [w for w in widgetParent.children()
                                if isinstance(w, PortConnection)]
                else:
                    children = []
                for w in children:
                    if w.sourcePort() == self.createSourcePort(w1, connection[1]):
                        col = w.colorIndex
                for w in children:
                    if w.sinkPort().parent() == w1:
                        col = w.colorIndex
                if col < 0:
                    self._colorIndex += 1
                    if self._colorIndex >= len(self._colors):
                        self._colorIndex = 0
                    col = self._colorIndex
                connectionWidget = self.createConnection(w1, connection[1], w2, connection[3], self._colors[col])
                connectionWidget.colorIndex = self._colorIndex
                connectionWidget.show()
