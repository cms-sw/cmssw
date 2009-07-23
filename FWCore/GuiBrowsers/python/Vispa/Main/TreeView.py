import logging

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Vispa.Main.BasicDataAccessor import *
from Vispa.Main.AbstractView import *

class TreeView(AbstractView, QTreeWidget):
    """ The TreeView widget fills itself using a DataAccessor.
    """
    def __init__(self, parent=None, maxDepth=100):
        logging.debug(__name__ + ": __init__")
        AbstractView.__init__(self)
        QTreeWidget.__init__(self, parent)

        self._operationId = 0
        self._firstItem = None
        self._itemDict = {}
        self._maxDepth = maxDepth
        self._selection = None
        self._updatingFlag = False

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
        AbstractView.setDataAccessor(self, accessor)

    def setDataObjects(self, objects):
        if len(self._dataObjects)!=len(objects):
            self._selection=None
        AbstractView.setDataObjects(self, objects)
    
    def clear(self):
        """ Deletes all items in the TreeView
        """
        logging.debug(__name__ + ": clear")
        self._operationId += 1
        self._itemDict = {}
        self._firstItem = None
        QTreeWidget.clear(self)

    def updateContent(self):
        """ Clear the TreeView and refill it.
        """
        logging.debug(__name__ + ": updateContent")
        self._updatingFlag = True
        self.clear()
        operationId = self._operationId
        if self._dataAccessor:
            i = 0
            for object in self._filter(self._dataObjects):
                # Process application event loop in order to accept user input during time consuming drawing operation
                QCoreApplication.instance().processEvents()
                # Abort drawing if operationId out of date
                if operationId != self._operationId:
                    return False
                self._createNode(object, self, str(i))
                i += 1
        self.expandToDepth(0)
        self._updatingFlag = False
        
    def _createNode(self, object=None, itemParent=None, positionName="0"):
        """ Create daughter items of an object recursively.
        """
        item = QTreeWidgetItem(itemParent)
        item.setText(0, self._dataAccessor.label(object))
        item.object = object
        item.positionName = str(positionName)+"("+self._dataAccessor.label(object)+")"
        self._itemDict[item.positionName] = item
        if self._firstItem==None:
            self._firstItem=item
        i = 0
        if len(positionName.split("-")) < self._maxDepth:
            for daughter in self._filter(self._dataAccessor.children(object)):
                self._createNode(daughter, item, positionName + "-" + str(i))
                i += 1

    def itemSelectionChanged(self):
        """ Emits signal selected that the TabController can connect to.
        """
        logging.debug(__name__ + ": itemSelectionChanged")
        if not self._updatingFlag:
            self._selection = self.currentItem().positionName
            self.emit(SIGNAL("selected"), self.currentItem().object)
        
    def select(self, object):
        """ Mark an object in the TreeView as selected.
        """
        logging.debug(__name__ + ": select")
        items = []
        for positionName, item in self._itemDict.items():
            if item.object == object:
                items += [(positionName, item)]
        if len(items) > 0:
            item = sorted(items)[0][1]
            self._selection = item.positionName
            self._updatingFlag = True
            self.setCurrentItem(item)
            self._updatingFlag = False

    def _selectedItem(self):
        if self._selection in self._itemDict.keys():
            return self._itemDict[self._selection]
        elif self._firstItem!=None:
            return self._firstItem
        else:
            return None

    def restoreSelection(self):
        """ Restore selection.
        """
        logging.debug(__name__ + ": restoreSelection")
        select = self._selectedItem()
        if select != None:
            self._updatingFlag = True
            self.setCurrentItem(select)
            self._updatingFlag = False
    
    def selection(self):
        """ Return currently selected object.
        
        If selection unknown return first object.
        """
        logging.debug(__name__ + ": selection")
        select = self._selectedItem()
        if select != None:
            return select.object
        else:
            return None
