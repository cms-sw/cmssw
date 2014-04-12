import logging

from PyQt4.QtCore import SIGNAL,Qt
from PyQt4.QtGui import QFrame,QHeaderView,QToolButton,QStandardItemModel,QVBoxLayout,QSizePolicy

from Vispa.Main.SplitterTab import SplitterTab
from Vispa.Gui.Header import FrameWithHeader
from Vispa.Gui.ZoomableScrollArea import ZoomableScrollArea
from Vispa.Gui.ZoomableScrollableWidgetOwner import ZoomableScrollableWidgetOwner
from Vispa.Gui.Zoomable import Zoomable
from Vispa.Views.AbstractView import NoneView
from Vispa.Views.TreeView import TreeView

class BrowserTab(SplitterTab):
    """ The BrowserTab has three views and is controlled by the BrowserControllerTab.
    """
    def __init__(self, parent=None, topLevelPropertyView=False):
        logging.debug(__name__ + ": __init__")
        SplitterTab.__init__(self, parent, topLevelPropertyView)

        self._treeView = None
        self._centerView = None

        self.createTreeView()
        self.createCenterView()
        self.createPropertyView()
        if topLevelPropertyView:
            self.horizontalSplitter().setSizes([300, 400])
            self.setSizes([700, 300])
        else:
            self.horizontalSplitter().setSizes([300, 400, 300])
        
    def createTreeView(self,parent=None):
        """ Create the tree view.
        """
        if not parent:
            parent=self.horizontalSplitter()
        
        self._treeviewArea = FrameWithHeader(parent)
        self._treeviewArea.header().setText("Tree View")
        self._treeviewArea.header().setToolTip("click on '>' for options of this view")
        self._treeViewMenuButton = self._treeviewArea.header().createMenuButton()
        self._treeView = TreeView(self._treeviewArea)
        self._treeviewArea.addWidget(self._treeView)
        
    def createCenterView(self,parent=None):
        """ Create the center view.
        """
        if not parent:
            parent=self.horizontalSplitter()
            
        self._centerArea = FrameWithHeader(parent)
        self._centerArea.header().setText("Center View")
        self._centerArea.header().setToolTip("click on '>' for options of this view")
        self._centerArea.header().createMenuButton()
        
        self._scrollArea=ZoomableScrollArea(self._centerArea)
        self._centerArea.addWidget(self._scrollArea)
        self.setCenterView(NoneView())
            
    def setCenterView(self,view):
        """ Set the center view.
        """
        logging.debug(self.__class__.__name__ +": setCenterView()")
        if self.centerView():
            self.centerView().close()
        self._scrollArea.takeWidget()
        self._centerView = view
        if isinstance(self.centerView(), ZoomableScrollableWidgetOwner):
            if isinstance(self.centerView(), Zoomable):
                self.centerView().setZoom(self._scrollArea.zoom())
            self._scrollArea.setWidget(self.centerView())
            self._scrollArea.show()
        else:
            self.centerView().resize(self._scrollArea.size())
            self._scrollArea.hide()
            self._centerArea.layout().addWidget(self.centerView())
            self.centerView().show()

    def treeView(self):
        return self._treeView
    
    def centerView(self):
        return self._centerView
    
    def setController(self, controller):
        """ Sets controller for this tab and connects signals etc.
        """
        SplitterTab.setController(self, controller)
        if self._scrollArea:
            self.connect(self._scrollArea, SIGNAL('wheelZoom()'), controller.resetZoomButtonPressedBefore)
            self.connect(self._scrollArea, SIGNAL("zoomChanged(float)"), controller.zoomChanged)
            self.connect(self.centerViewMenuButton(), SIGNAL("clicked(bool)"), controller.centerViewMenuButtonClicked)
            self.connect(self.treeViewMenuButton(), SIGNAL("clicked(bool)"), controller.treeViewMenuButtonClicked)

    def scrollArea(self):
        return self._scrollArea
    
    def treeViewMenuButton(self):
        return self._treeviewArea.header().menuButton()
    
    def centerViewMenuButton(self):
        return self._centerArea.header().menuButton()
    
    def treeViewHeader(self):
        return self._treeviewArea.header()
    
    def centerViewHeader(self):
        return self._centerArea.header()
    
    def setCenterViewHeader(self,text):
        self._centerArea.header().setText(text)

    def setTreeViewHeader(self,text):
        self._treeviewArea.header().setText(text)
