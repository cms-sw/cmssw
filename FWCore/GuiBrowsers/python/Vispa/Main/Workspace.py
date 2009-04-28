import logging

from PyQt4.QtGui import *

from Vispa.Main.ZoomableWidget import *
from Vispa.Main.ZoomableScrollArea import *
from Vispa.Main.ConnectableWidgetOwner import *

class Workspace(ZoomableWidget, ConnectableWidgetOwner):
    """ Area for drawing connectable widgets.
    
    The Workspace maps all widgets to an id and a data object.
    New widgets have to be added by using the addWidget function.
    A widget can be retrieved by widgetById or widgetByObject.
    """
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        ZoomableWidget.__init__(self, parent)
        ConnectableWidgetOwner.__init__(self)

        self.widgetDict = {}

    def setZoom(self, zoom):
        """ Sets zoom.
        """
        ZoomableWidget.setZoom(self, zoom)
        # make sure connections are updated after widgets were moved
        # setZoom function of ZoomableWidget does not guarantee connections are updated after widgets
        self.updateConnections()
        
    def autosizeScrollArea(self):
        """ If this window is widget of a ZoomableScrollArea tell scroll area to autosize.
        """
        if self.parent() and isinstance(self.parent().parent(), ZoomableScrollArea):
            # Why parent().parent()?
            # parent() is QScrollArea.viewport(), basically the QScrollArea without scroll bars
            # parent().parent() is eventually the QScrollArea
            self.parent().parent().autosizeScrollWidget()
    
    def widgetMoved(self, widget):
        """ Calls autosizeScrollArea().
        """
        ConnectableWidgetOwner.widgetMoved(self, widget)
        self.autosizeScrollArea()

    def widgetSelected(self, widget):
        """ Emits signal widgetSelected that the TabController can connect to.
        """
        logging.debug(__name__ + ": widgetSelected")
        ConnectableWidgetOwner.widgetSelected(self, widget)
        self.emit(SIGNAL("widgetSelected"), widget)
        
    def deselectAllWidgets(self, exception=None):
        ConnectableWidgetOwner.deselectAllWidgets(self,exception)
        self.emit(SIGNAL("widgetSelected"), None)

    def select(self, widget):
        """ Mark a widget as selected.
        """
        if widget:
            widget.select()
        else:
            self.deselectAllWidgets()

    def widgetById(self, id):
        """ Return a widget in the Workspace with a certain id.
        
        The id is unique inside the Workspace and can be accessed by anyWidget.widgetId.
        """
        if id in self.widgetDict.keys():
            return self.widgetDict[id]
        return None

    def widgetByObject(self, object):
        """ Return a widget in the Workspace with a certain object.
        """
        widgets = []
        for id, widget in self.widgetDict.items():
            if widget.object == object:
                widgets += [(id, widget)]
        if len(widgets) > 0:
            return sorted(widgets)[0][1]
        return None

    def clear(self):
        """ Deletes all boxes in the Workspace
        """
        logging.debug(__name__ + ": clear")
        self.widgetDict = {}
        for w in self.children():
            w.setParent(None)
            w.deleteLater()
            
    def addWidget(self, widget, object=None, id=0):
        if id in self.widgetDict.keys():
            id = 0
            while id in self.widgetDict.keys():
                id += 1
        widget.widgetId = id
        widget.object = object
        self.widgetDict[widget.widgetId] = widget

    def closeEvent(self, event):
        self.clear()
        event.accept()

    def updateContent(self):
        pass

    def setDataObjects(self, objects):
        pass

    def setDataAccessor(self, accessor):
        pass

    def setBoxContentScript(self, script):
        pass
