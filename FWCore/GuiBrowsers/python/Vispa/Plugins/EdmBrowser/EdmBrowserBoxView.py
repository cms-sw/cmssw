import logging

from Vispa.Views.BoxDecayView import BoxDecayView
from Vispa.Gui.WidgetContainer import WidgetContainer

class EdmBrowserBoxView(BoxDecayView):
    """
    """
    LABEL="BoxView"
    
    def createBox(self, widgetParent, container, title, text):
        widget=BoxDecayView.createBox(self, widgetParent, container, title, text)
        if isinstance(widget,WidgetContainer):
            widget.setNotCollapsable()
        return widget

    def selection(self):
        return self.dataAccessor().read(BoxDecayView.selection(self))

    def isUpdated(self,object):
        widget=self.widgetByObject(object)
        if widget:
            return len(self.dataAccessor().children(object))==len(widget.children()) and\
                ((isinstance(widget,WidgetContainer) and len(self.dataAccessor().children(object))>0) or\
                 (not isinstance(widget,WidgetContainer) and len(self.dataAccessor().children(object))==0))
        else:
            return False
