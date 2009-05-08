import logging

from PyQt4.QtGui import *

from Vispa.Main.AbstractView import *
from Vispa.Main.ZoomableWidget import *
from Vispa.Main.ZoomableScrollArea import *
from Vispa.Main.ConnectableWidgetOwner import *

class Workspace(AbstractView, ZoomableWidget, ConnectableWidgetOwner):
    """ Area for drawing connectable widgets.
    
    New widgets have to be added by using the addWidget function.
    The Workspace takes of the selection of widgets.
    """
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        AbstractView.__init__(self)
        ZoomableWidget.__init__(self, parent)
        ConnectableWidgetOwner.__init__(self)

        self._widgetDict = {}
        self._selection = None

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
        if not self._updatingFlag:
            self._selection = widget.widgetId
            self.emit(SIGNAL("selected"), widget.object)
        
    def deselectAllWidgets(self, exception=None):
        """ Emits signal widgetSelected that the TabController can connect to.
        """
        ConnectableWidgetOwner.deselectAllWidgets(self, exception)
        if not self._updatingFlag:
            self._selection = None
            self.emit(SIGNAL("selected"), None)

    def _widgetByObject(self, object):
        widgets = []
        for id, widget in self._widgetDict.items():
            if widget.object == object:
                widgets += [(id, widget)]
        if len(widgets) > 0:
            return sorted(widgets)[0][1]
        else:
            return None

    def select(self, object):
        """ Mark an object as selected.
        """
        widget = self._widgetByObject(object)
        if widget!=None:
            self._selection = widget.widgetId
            self._updatingFlag = True
            widget.select()
            self._updatingFlag = False
        else:
            self._selection = None
            self._updatingFlag = True
            self.deselectAllWidgets()
            self._updatingFlag = False

    def restoreSelection(self):
        """ Restore selection.
        """
        if self._selection in self._widgetDict.keys():
            widget = self._widgetDict[self._selection]
            widget.select()

    def selection(self):
        """ Return the currently selected object.
        """
        if self._selection in self._widgetDict.keys():
            return self._widgetDict[self._selection].object
        else:
            return None

    def clear(self):
        """ Deletes all boxes in the Workspace
        """
        logging.debug(__name__ + ": clear")
        self._widgetDict = {}
        for w in self.children():
            w.setParent(None)
            w.deleteLater()
            
    def addWidget(self, widget, object=None, id=0):
        """ Add widget to the view and map it to an id.
        """
        if id in self._widgetDict.keys():
            id = 0
            while id in self._widgetDict.keys():
                id += 1
        widget.widgetId = str(id)+"("+self._dataAccessor.label(object)+")"
        widget.object = object
        self._widgetDict[widget.widgetId] = widget

    def closeEvent(self, event):
        self.clear()
        event.accept()
