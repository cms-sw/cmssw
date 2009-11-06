import logging

from PyQt4.QtCore import SIGNAL,Qt
from PyQt4.QtGui import QFrame,QHeaderView,QToolButton,QStandardItemModel,QVBoxLayout,QSizePolicy

from Vispa.Main.SplitterTab import SplitterTab
from Vispa.Gui.ZoomableScrollArea import ZoomableScrollArea
from Vispa.Gui.ZoomableScrollableWidgetOwner import ZoomableScrollableWidgetOwner
from Vispa.Gui.Zoomable import Zoomable
from Vispa.Views.AbstractView import NoneView
from Vispa.Views.TreeView import TreeView

class HeaderView(QHeaderView):
    def mousePressEvent(self,event):
        QHeaderView.mousePressEvent(self,event)
        if event.button()==Qt.RightButton:
            self.emit(SIGNAL("mouseRightPressed"), event.globalPos())

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
        self._treeviewArea = QFrame(parent)
        self._treeviewArea.setFrameShadow(QFrame.Raised)
        self._treeviewArea.setFrameStyle(QFrame.StyledPanel)
        self._treeviewArea.setLayout(QVBoxLayout())
        self._treeviewArea.layout().setContentsMargins(0, 0, 0, 0)
        self._treeviewArea.layout().setSpacing(0)
        self._treeViewHeaderModel = QStandardItemModel(self._treeviewArea)
        self.setTreeViewHeader("Tree View")
        self._treeViewHeader = HeaderView(Qt.Horizontal, self._treeviewArea)
        self._treeViewHeader.setModel(self._treeViewHeaderModel)
        self._treeViewHeader.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._treeViewHeader.setStretchLastSection(True)
        self._treeViewHeader.setFixedHeight(25)
        self._treeViewMenuButton=QToolButton(self._treeViewHeader)
        self._treeViewMenuButton.setText(">")
        self._treeviewArea.layout().addWidget(self._treeViewHeader)
        self._treeView = TreeView(self._treeviewArea)
        self._treeView.setFrameStyle(QFrame.NoFrame)
        self._treeviewArea.layout().addWidget(self._treeView)

    def createCenterView(self,parent=None):
        """ Create the center view.
        """
        if not parent:
            parent=self.horizontalSplitter()
        self._centerArea = QFrame(parent)
        self._centerArea.setFrameShadow(QFrame.Raised)
        self._centerArea.setFrameStyle(QFrame.StyledPanel)
        self._centerArea.setLayout(QVBoxLayout())
        self._centerArea.layout().setContentsMargins(0, 0, 0, 0)
        self._centerArea.layout().setSpacing(0)
        self._centerViewHeaderModel = QStandardItemModel(self._centerArea)
        self.setCenterViewHeader("Center View")
        self._centerHeader = HeaderView(Qt.Horizontal, self._centerArea)
        self._centerHeader.setModel(self._centerViewHeaderModel)
        self._centerHeader.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._centerHeader.setStretchLastSection(True)
        self._centerHeader.setFixedHeight(25)
        self._centerViewMenuButton=QToolButton(self._centerHeader)
        self._centerViewMenuButton.setText(">")
        self._centerArea.layout().addWidget(self._centerHeader)
        self._scrollArea=ZoomableScrollArea(self._centerArea)
        self._scrollArea.setFrameStyle(QFrame.NoFrame)
        self._centerArea.layout().addWidget(self._scrollArea)
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
            self.connect(self._centerViewMenuButton, SIGNAL("clicked(bool)"), controller.centerViewMenuButtonClicked)
            self.connect(self._treeViewMenuButton, SIGNAL("clicked(bool)"), controller.treeViewMenuButtonClicked)

    def scrollArea(self):
        return self._scrollArea
    
    def treeViewMenuButton(self):
        return self._treeViewMenuButton
    
    def centerViewMenuButton(self):
        return self._centerViewMenuButton
    
    def treeViewHeader(self):
        return self._treeViewHeader
    
    def centerViewHeader(self):
        return self._centerHeader
    
    def setCenterViewHeader(self,text):
        self._centerViewHeaderModel.setHorizontalHeaderLabels([text])

    def setTreeViewHeader(self,text):
        self._treeViewHeaderModel.setHorizontalHeaderLabels([text])
