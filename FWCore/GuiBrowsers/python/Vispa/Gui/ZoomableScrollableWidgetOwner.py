import logging

from PyQt4.QtGui import *

from Vispa.Gui.ZoomableWidget import ZoomableWidget
from Vispa.Gui.ZoomableScrollArea import ZoomableScrollArea
from Vispa.Gui.ConnectableWidgetOwner import ConnectableWidgetOwner

class ZoomableScrollableWidgetOwner(ConnectableWidgetOwner, ZoomableWidget):
    """ Area for drawing connectable widgets.
    """
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        ZoomableWidget.__init__(self, parent)
        ConnectableWidgetOwner.__init__(self)

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
    
    def widgetDragged(self, widget):
        """ Calls autosizeScrollArea().
        """
        ConnectableWidgetOwner.widgetDragged(self, widget)
        self.autosizeScrollArea()
