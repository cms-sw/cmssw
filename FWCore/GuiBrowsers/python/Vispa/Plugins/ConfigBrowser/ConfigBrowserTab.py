import logging

from PyQt4.QtCore import SIGNAL

from Vispa.Main.SplitterTab import SplitterTab
from Vispa.Main.TreeView import TreeView
from Vispa.Main.ZoomableScrollArea import ZoomableScrollArea
from Vispa.Main.WidgetView import WidgetView
from ConfigBrowserBoxView import ConfigBrowserBoxView

class ConfigBrowserTab(SplitterTab):
    """
    """
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        SplitterTab.__init__(self, parent)

        self._treeView = None
        self._centerView = None

        self._createTreeView()
        self._createCenterView()
        self.createPropertyView()
        self.setSizes([300, 400, 300])
        
    def _createTreeView(self):
        self._treeView = TreeView(self)
    
    def _createCenterView(self):
        """ Create the center view.
        """
        self._scrollArea = ZoomableScrollArea(self)
        self._centerView = WidgetView() 
        self._scrollArea.setWidget(self._centerView)

    def setCenterView(self, view):
        """ Create the center view.
        """
        self._centerView = view
        self._scrollArea.takeWidget()
        self._scrollArea.setWidget(self._centerView)

    def setController(self, controller):
        """ Sets controller for this tab and connects signals etc.
        """
        SplitterTab.setController(self, controller)

        self.connect(self.propertyView(), SIGNAL('valueChanged()'), controller.setModified)
        self.connect(self._scrollArea, SIGNAL('wheelZoom()'), controller.resetZoomButtonPressedBefore)
        self.connect(self._scrollArea, SIGNAL("zoomChanged(float)"), controller.zoomChanged)
                
    def scrollArea(self):
        return self._scrollArea
        
    def treeView(self):
        return self._treeView
    
    def centerView(self):
        return self._centerView
    
    def selected(self):
        self.controller().selected()

    def closeEvent(self, event):
        self.controller().closeEvent(event)
