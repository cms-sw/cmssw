from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import QSplitter, QColor, QPalette

from Vispa.Main.AbstractTab import AbstractTab
from Vispa.Main.PropertyView import PropertyView
from Vispa.Main.SimpleDragableTreeWidget import SimpleDragableTreeWidget
from Vispa.Main.ZoomableScrollArea import ZoomableScrollArea

class SplitterTab(QSplitter, AbstractTab):
    """ A Tab with a QSplitter and a function to create the PropertyView.
    """
    def __init__(self, parent=None):
        AbstractTab.__init__(self)
        QSplitter.__init__(self, parent)
        
        self.adjustSize()
        self.resize(self.childrenRect().size())
        self._scrollArea = None
        self._propertyView = None
        self._lightBlueBackgroundColor = None
        self._dragableLists = {}
        
    def setController(self, controller):
        AbstractTab.setController(self, controller)
        if self._scrollArea:
            self.connect(self._scrollArea, SIGNAL('wheelZoom()'), controller.resetZoomButtonPressedBefore)
            self.connect(self._scrollArea, SIGNAL("zoomChanged(float)"), controller.zoomChanged)
        
    def createScrollArea(self):
        """ Creates ZoomableScrollArea object, adds it to this tab and makes it available via zoomableScrollArea().
        """
        self._scrollArea = ZoomableScrollArea(self)
    
    def scrollArea(self):
        """ Returns scroll area of this tab.
        """
        return self._scrollArea
    
    def createPropertyView(self):
        """ Creates PropertyView object, adds it to this tab and makes it available via propertyView().
        """
        self._propertyView = PropertyView(self, "PropertyView")
        
    def propertyView(self):
        """ Returns PropertyView object. See createPropertyView().
        """
        return self._propertyView
    
    def lightBlueBackgroundColor(self):
        """ Returns a QColor object suitable as discrete background color.
        """
        if not self._lightBlueBackgroundColor:
            self._lightBlueBackgroundColor = QColor(Qt.blue).lighter(195)
        return self._lightBlueBackgroundColor
    
    def createDragableList(self, id, headerLabel, dragEnabled=False, dragMimeType=None):
        """ Creates a SimpleDragableTreeWidget object and stores it in a dictionary. It can be referenced by its id.
        
        The list is a QTreeWidget object with one column.
        The list is not added to this tab, so it can be used in other widgets as well.
        """
        self._dragableLists[id] = SimpleDragableTreeWidget(dragMimeType)
        self._dragableLists[id].setColumnCount(1)
        self._dragableLists[id].setHeaderLabels([headerLabel])
        self._dragableLists[id].palette().setColor(QPalette.Base, self.lightBlueBackgroundColor())       # OS X
        self._dragableLists[id].palette().setColor(QPalette.Window, self.lightBlueBackgroundColor())
        if dragEnabled:
            self._dragableLists[id].setDragEnabled(True)
            
    def dragableList(self, id):
        """ Returns dragableList created with createDraggableList() with given id.
        
        If there is no such list None will be returned.
        """
        if id in self._dragableLists.keys():
            return self._dragableLists[id]
        return None
    
    def populateDragableList(self, id, items):
        """ Fills items into dragable list with given id.
        """
        if id in self._dragableLists.keys():
            self._dragableLists[id].insertTopLevelItems(0, items)
