import logging

from PyQt4.QtCore import SIGNAL,Qt
from PyQt4.QtGui import QWidget

from Vispa.Views.AbstractView import AbstractView
from Vispa.Gui.ZoomableScrollableWidgetOwner import ZoomableScrollableWidgetOwner
from Vispa.Gui.ZoomableScrollArea import ZoomableScrollArea

class WidgetView(AbstractView, ZoomableScrollableWidgetOwner):
    """ Area for drawing connectable widgets.
    
    New widgets have to be added by using the addWidget function.
    The WidgetView takes of the selection of widgets.
    """
    
    LABEL = "&Widget View"
    
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        AbstractView.__init__(self)
        ZoomableScrollableWidgetOwner.__init__(self, parent)

        self._widgetDict = {}
        self._selection = None
        self._updatingFlag = 0

    def widgets(self):
        return self._widgetDict.values()

    def widgetSelected(self, widget, multiSelect=False):
        """ Emits signal widgetSelected that the TabController can connect to.
        """
        logging.debug(__name__ + ": widgetSelected")
        ZoomableScrollableWidgetOwner.widgetSelected(self, widget, multiSelect)
        if not self._updatingFlag and hasattr(widget, "object"):
            if hasattr(widget, "positionName"):
                self._selection = widget.positionName
                self.emit(SIGNAL("selected"), widget.object)
            else:
                self.emit(SIGNAL("selected"), widget.object())
        
    def deselectAllWidgets(self, exception=None):
        """ Deselect all widget in view.
        """
        ZoomableScrollableWidgetOwner.deselectAllWidgets(self, exception)
        if not self._updatingFlag:
            self._selection = None
        if not exception:
            self.emit(SIGNAL("selected"), None)

    def widgetByObject(self, object):
        widgets = []
        for positionName, widget in self._widgetDict.items():
            if widget.object == object:
                widgets += [(positionName, widget)]
        if len(widgets) > 0:
            return sorted(widgets)[0][1]
        else:
            return None

    def select(self, object, offset=5):
        """ Mark an object as selected.
        """
        widget = self.widgetByObject(object)
        if widget!=None:
            self._selection = widget.positionName
            self._updatingFlag +=1
            widget.select()
            if self.parent() and isinstance(self.parent().parent(), ZoomableScrollArea):
                self.parent().parent().ensureWidgetVisible(widget,offset,offset)
            self._updatingFlag -=1
        else:
            self._selection = None
            self._updatingFlag +=1
            self.deselectAllWidgets()
            self._updatingFlag -=1

    def restoreSelection(self,offset=5):
        """ Restore selection.
        """
        if self._selection in self._widgetDict.keys():
            widget = self._widgetDict[self._selection]
            self._updatingFlag +=1
            widget.select()
            if self.parent() and isinstance(self.parent().parent(), ZoomableScrollArea):
                self.parent().parent().ensureWidgetVisible(widget,offset,offset)
            self._updatingFlag -=1

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
        # logging statement here can be very useful for debugging / improving performance
        logging.debug(__name__ + ": clear")
        self._widgetDict = {}
        for w in self.children():
            if isinstance(w,QWidget):
                w.setParent(None)
                w.deleteLater()
            
    def addWidget(self, widget, object=None, positionName=0):
        """ Add widget to the view and map it to an id.
        """
        if str(positionName)+"("+self.dataAccessor().label(object)+")" in self._widgetDict.keys():
            positionName = 0
            while positionName in self._widgetDict.keys():
                positionName += 1
        widget.positionName = str(positionName)+"("+self.dataAccessor().label(object)+")"
        widget.object = object
        self._widgetDict[widget.positionName] = widget

    def closeEvent(self, event):
        self.clear()
        event.accept()

    def setDataObjects(self, objects):
        AbstractView.setDataObjects(self, objects)
        self.clear()

    def mousePressEvent(self,event):
        if event.button()==Qt.RightButton:
            self.emit(SIGNAL("mouseRightPressed"), event.globalPos())
        ZoomableScrollableWidgetOwner.mousePressEvent(self,event)

    def isBusy(self):
        return self._updatingFlag>0

    def widgetDoubleClicked(self, widget):
        """ Emits signal doubleClicked that the TabController can connect to.
        """
        logging.debug(__name__ + ": widgetDoubleClicked")
        ZoomableScrollableWidgetOwner.widgetDoubleClicked(self, widget)
        if hasattr(widget, "object"):
            if hasattr(widget, "positionName"):
                self._selection = widget.positionName
                self.emit(SIGNAL("doubleClicked"), widget.object)
            else:
                self.emit(SIGNAL("doubleClicked"), widget.object())
