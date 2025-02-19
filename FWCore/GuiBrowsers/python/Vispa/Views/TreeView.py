import logging

from PyQt4.QtCore import Qt,SIGNAL,QCoreApplication
from PyQt4.QtGui import QTreeWidget,QTreeWidgetItem,QInputDialog

from Vispa.Main.Application import Application
from Vispa.Share.BasicDataAccessor import BasicDataAccessor
from Vispa.Views.AbstractView import AbstractView

class TreeView(AbstractView, QTreeWidget):
    """ The TreeView widget fills itself using a DataAccessor.
    """

    LABEL = "&Tree View"
    UPDATE_EVERY = 20
    
    def __init__(self, parent=None, maxDepth=100):
        logging.debug(__name__ + ": __init__")
        AbstractView.__init__(self)
        QTreeWidget.__init__(self, parent)

        self._operationId = 0
        self._firstItem = None
        self._itemDict = {}
        self._maxDepth = maxDepth
        self._selection = None
        self._updatingFlag = 0
        self._treeDepth=1
        self._updateCounter=0

        self.setSortingEnabled(False)
        self.setColumnCount(1)
        self.setRootIsDecorated(True)
        self.header().hide()

        self.connect(self, SIGNAL("itemSelectionChanged()"), self.itemSelectionChanged)
        self.connect(self, SIGNAL("itemCollapsed(QTreeWidgetItem*)"), self.itemExpanded)
        self.connect(self, SIGNAL("itemExpanded(QTreeWidgetItem*)"), self.itemExpanded)
        
    def setDataAccessor(self, accessor):
        """ Sets the DataAccessor from which the nodes are created.
        
        You need to call updateContent() in order to make the changes visible.   
        """
        if not isinstance(accessor, BasicDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type BasicDataAccessor.")
        AbstractView.setDataAccessor(self, accessor)

    def cancel(self):
        """ Stop all running operations.
        """
        self._operationId += 1
        
    def clear(self):
        """ Deletes all items in the TreeView
        """
        #logging.debug(__name__ + ": clear")
        self._itemDict = {}
        self._firstItem = None
        QTreeWidget.clear(self)

    def updateContent(self):
        """ Clear the TreeView and refill it.
        """
        #logging.debug(__name__ + ": updateContent")
        self.cancel()
        if self.dataAccessor() == None:
            return False
        self._updatingFlag+=1
        self.clear()
        if self.dataObject()==None:
            self._updatingFlag-=1
            return True
        operationId = self._operationId
        i = 0
        for object in self.applyFilter(self.dataObjects()):
            # Process application event loop in order to accept user input during time consuming drawing operation
            self._updateCounter+=1
            if self._updateCounter>=self.UPDATE_EVERY:
                self._updateCounter=0
                if not Application.NO_PROCESS_EVENTS:
                    QCoreApplication.instance().processEvents()
            # Abort drawing if operationId out of date
            if operationId != self._operationId:
                break
            self._createNode(object, self, str(i))
            i += 1
        if self._treeDepth>0:
            self.expandToDepth(self._treeDepth-1)
        self._updatingFlag -=1
        return self._operationId==operationId
        
    def _createNode(self, object=None, itemParent=None, positionName="0"):
        """ Create daughter items of an object recursively.
        """
        item = QTreeWidgetItem(itemParent)
        item.setText(0, self.dataAccessor().label(object))
        item.object = object
        item.positionName = str(positionName)+"("+self.dataAccessor().label(object)+")"
        self._itemDict[item.positionName] = item
        if self._firstItem==None:
            self._firstItem=item
        i = 0
        if len(positionName.split("-")) < self._maxDepth:
            for daughter in self.applyFilter(self.dataAccessor().children(object)):
                self._createNode(daughter, item, positionName + "-" + str(i))
                i += 1

    def itemSelectionChanged(self):
        """ Emits signal selected that the TabController can connect to.
        """
        #logging.debug(__name__ + ": itemSelectionChanged")
        if not self._updatingFlag:
            self._selection = self.currentItem().positionName
            self.emit(SIGNAL("selected"), self.currentItem().object)
        
    def select(self, object):
        """ Mark an object in the TreeView as selected.
        """
        #logging.debug(__name__ + ": select")
        items = []
        for positionName, item in self._itemDict.items():
            if item.object == object:
                items += [(positionName, item)]
        if len(items) > 0:
            item = sorted(items)[0][1]
            self._selection = item.positionName
            self._updatingFlag +=1
            self.setCurrentItem(item)
            self._updatingFlag -=1

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
        #logging.debug(__name__ + ": restoreSelection")
        select = self._selectedItem()
        if select != None:
            self._updatingFlag +=1
            self.setCurrentItem(select)
            self._updatingFlag -=1
    
    def selection(self):
        """ Return currently selected object.
        
        If selection unknown return first object.
        """
        #logging.debug(__name__ + ": selection")
        select = self._selectedItem()
        if select != None:
            return select.object
        else:
            return None

    def isBusy(self):
        return self._updatingFlag>0

    def mousePressEvent(self,event):
        QTreeWidget.mousePressEvent(self,event)
        if event.button()==Qt.RightButton:
            self.emit(SIGNAL("mouseRightPressed"), event.globalPos())

    def expandToDepthDialog(self):
        """ Show dialog and call expandToDepth() function of tree view.
        """
        if hasattr(QInputDialog, "getInteger"):
            # Qt 4.3
            (depth, ok) = QInputDialog.getInteger(self, "Expand to depth...", "Input depth:", self._treeDepth, 0)
        else:
            # Qt 4.5
            (depth, ok) = QInputDialog.getInt(self, "Expand to depth...", "Input depth:", self._treeDepth, 0)
        if ok:
            self._treeDepth=depth
            self.collapseAll(False)
            if self._treeDepth>0:
                self.expandToDepth(self._treeDepth-1)

    def expandAll(self):
        self._treeDepth=10000
        QTreeWidget.expandAll(self)
        
    def collapseAll(self,remember=True):
        if remember:
            self._treeDepth=0
        QTreeWidget.collapseAll(self)
        
    def itemExpanded(self,item):
        i=0
        while item:
            if item.isExpanded():
                i+=1
            item=item.parent()
        self._treeDepth=i
        