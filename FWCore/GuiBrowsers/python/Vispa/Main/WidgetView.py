import logging

from PyQt4.QtGui import *

from Vispa.Main.AbstractView import *
from Vispa.Main.ZoomableScrollableWidgetOwner import *

class WidgetView(AbstractView, ZoomableScrollableWidgetOwner):
    """ Area for drawing connectable widgets.
    
    New widgets have to be added by using the addWidget function.
    The WidgetView takes of the selection of widgets.
    """
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        AbstractView.__init__(self)
        ZoomableScrollableWidgetOwner.__init__(self, parent)

        self._widgetDict = {}
        self._selection = None
        self._updatingFlag = False

    def widgetSelected(self, widget):
        """ Emits signal widgetSelected that the TabController can connect to.
        """
        logging.debug(__name__ + ": widgetSelected")
        ZoomableScrollableWidgetOwner.widgetSelected(self, widget)
        if not self._updatingFlag:
            self._selection = widget.positionName
            self.emit(SIGNAL("selected"), widget.object)
        
    def deselectAllWidgets(self, exception=None):
        """ Emits signal widgetSelected that the TabController can connect to.
        """
        ZoomableScrollableWidgetOwner.deselectAllWidgets(self, exception)
        if not self._updatingFlag:
            self._selection = None
            self.emit(SIGNAL("selected"), None)

    def _widgetByObject(self, object):
        widgets = []
        for positionName, widget in self._widgetDict.items():
            if widget.object == object:
                widgets += [(positionName, widget)]
        if len(widgets) > 0:
            return sorted(widgets)[0][1]
        else:
            return None

    def select(self, object):
        """ Mark an object as selected.
        """
        widget = self._widgetByObject(object)
        if widget!=None:
            self._selection = widget.positionName
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
        """ Deletes all boxes in the WidgetView
        """
        logging.debug(__name__ + ": clear")
        self._widgetDict = {}
        for w in self.children():
            if isinstance(w,QWidget):
                w.setParent(None)
                w.deleteLater()
            
    def addWidget(self, widget, object=None, positionName=0):
        """ Add widget to the view and map it to an id.
        """
        if positionName in self._widgetDict.keys():
            positionName = 0
            while positionName in self._widgetDict.keys():
                positionName += 1
        widget.positionName = str(positionName)+"("+self._dataAccessor.label(object)+")"
        widget.object = object
        self._widgetDict[widget.positionName] = widget

    def closeEvent(self, event):
        self.clear()
        event.accept()
