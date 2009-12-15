import logging

from Vispa.Views.BoxDecayView import BoxDecayView

class EdmBrowserBoxView(BoxDecayView):
    """
    """
    LABEL="BoxView"
    
    def createBox(self, widgetParent, container, title, text):
        widget=BoxDecayView.createBox(self, widgetParent, container, title, text)
        widget.setToolTip("Double click to expand object")
        return widget
