import logging

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Vispa.Main.BasicDataAccessor import *

class TreeView(QTreeWidget):
    """ The TreeView widget fills itself using a DataAccessor.
    """
    def __init__(self, parent=None, name=None, fl=0):
        logging.debug(__name__ + ": __init__")
        QTreeWidget.__init__(self, parent)
        self._accessor = None
        self._itemDict = {}
        self._dataObjects = []

        self.setSortingEnabled(False)
        self.setColumnCount(1)
        self.setRootIsDecorated(True)
        self.header().hide()
        self.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding))
        
        self.connect(self, SIGNAL("itemSelectionChanged()"), self.itemSelectionChanged)
        
    def setDataAccessor(self, accessor):
        """ Sets the DataAccessor from which the nodes are created.
        
        You need to call updateContent() in order to make the changes visible.   
        """
        if not isinstance(accessor, BasicDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type BasicDataAccessor.")
        self._accessor = accessor
    
    def accessor(self):
        return self._accessor
    
    def setDataObjects(self, objects):
        """ Sets the objects that shall be shown.
        
        You need to call updateContent() in order to make the changes visible.   
        """
        self._dataObjects = objects
        
    def dataObjects(self):
        return self._dataObjects

    def clear(self):
        """ Deletes all items in the TreeView
        """
        logging.debug(__name__ + ": clear")
        self._itemDict = {}
        QTreeWidget.clear(self)

    def updateContent(self):
        """ Clear the TreeView and refill it.
        """
        self.clear()
        if self._accessor:
            i = 0
            for object in self._dataObjects:
                self._createNode(object, self, str(i))
                i += 1
        self.expandToDepth(0)
        
    def _createNode(self, object=None, itemParent=None, id="0"):
        """ Create daughter items of an object recursively.
        """
        item = QTreeWidgetItem(itemParent)
        item.setText(0, self._accessor.label(object))
        item.object = object
        item.itemId = id
        self._itemDict[item.itemId] = item
        i = 0
        for daughter in self._accessor.children(object):
            self._createNode(daughter, item, id + "-" + str(i))
            i += 1

    def itemSelectionChanged(self):
        """ Emits signal itemSelected that the TabController can connect to.
        """
        logging.debug(__name__ + ": itemSelected")
        self.emit(SIGNAL("itemSelected"), self.currentItem())
        
    def itemById(self, id):
        """ Return an item in the TreeView with a certain id.
        
        The id is unique inside the TreeView and can be accessed by anyItem.itemId.
        """
        if id in self._itemDict.keys():
            return self._itemDict[id]
        return None

    def select(self, item):
        """ Mark an item in the TreeView as selected.
        """
        if item != None:
            self.setCurrentItem(item)

    def itemByObject(self, object):
        """ Return an item in the TreeView with a certain object.
        """
        items = []
        for id, item in self._itemDict.items():
            if item.object == object:
                items += [(id, item)]
        if len(items) > 0:
            return sorted(items)[0][1]
        else:
            return None
