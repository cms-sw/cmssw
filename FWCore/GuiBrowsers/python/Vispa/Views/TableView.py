import logging

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Vispa.Main.Application import Application
from Vispa.Share.BasicDataAccessor import BasicDataAccessor
from Vispa.Views.AbstractView import AbstractView
from Vispa.Views.PropertyView import PropertyView,Property

class TableView(AbstractView, QTableWidget):
    """ The TableView widget fills itself using a DataAccessor.
    """
    
    LABEL = "&Table View"
    
    def __init__(self, parent=None, maxDepth=100):
        logging.debug(__name__ + ": __init__")
        AbstractView.__init__(self)
        QTableWidget.__init__(self, parent)

        self._operationId = 0
        self._selection = None
        self._updatingFlag = 0
        self._maxDepth=maxDepth
        self._columns=[]
        self._sortingFlag=False
        self._filteredColumns=[]
        self._firstColumn=0

        self.setSortingEnabled(False)
        self.verticalHeader().hide()
        self.setSelectionMode(QTableWidget.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding))

        self.connect(self, SIGNAL("itemSelectionChanged()"), self.itemSelectionChanged)

    def setSorting(self,sort):
        self._sortingFlag=sort
        
    def setColumns(self,columns):
        """ Set a list of columns that shall be shown.
        """
        self._filteredColumns=columns
        
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
    
    def cancel(self):
        """ Stop all running operations.
        """
        self._operationId += 1
        
    def clear(self):
        """ Deletes all items in the TableView
        """
        #logging.debug(__name__ + ": clear")
        QTableWidget.clear(self)
        self.setRowCount(0)
        self.setSortingEnabled(False)
        self._columns=[]

    def updateContent(self):
        """ Clear the TableView and refill it.
        """
        logging.debug(__name__ + ": updateContent")
        self.cancel()
        if self.dataAccessor() == None:
            return False
        self._updatingFlag+=1
        self.clear()
        if self.dataObject()==None:
            self._updatingFlag-=1
            return True
        operationId = self._operationId
        objects=[]
        for object in self.applyFilter(self.dataObjects()):
            objects+=self._getObjects(object)
        if self._filteredColumns!=[]:
            self._columns=self._filteredColumns
        else:
            self._columns=[]
            ranking={}
            for object,depth in objects:
                for property in self.dataAccessor().properties(object):
                    if not property[1] in ranking.keys():
                        ranking[property[1]]=1
                        if property[0]!="Category":
                            self._columns+=[property[1]]
                    elif property[1]=="Label":
                        ranking[property[1]]+=100000
                    elif property[1]=="Name":
                        ranking[property[1]]+=10000
                    else:
                        ranking[property[1]]+=1
            self._columns.sort(lambda x,y: cmp(-ranking[x],-ranking[y]))
        self.setColumnCount(len(self._columns))
        self.setHorizontalHeaderLabels(self._columns)
        for object,depth in objects:
            # Process application event loop in order to accept user input during time consuming drawing operation
            if not Application.NO_PROCESS_EVENTS:
                QCoreApplication.instance().processEvents()
            # Abort drawing if operationId out of date
            if operationId != self._operationId:
                break
            self._createItem(object,depth)
        for i in range(len(self._columns)):
            self.resizeColumnToContents(i)
        self.setSortingEnabled(self._sortingFlag)
        self._updatingFlag-=1
        return self._operationId==operationId

    def _getObjects(self, object, depth=0):
        objects=[(object,depth)]
        if depth < self._maxDepth:
            for daughter in self.applyFilter(self.dataAccessor().children(object)):
                objects+=self._getObjects(daughter,depth+1)
        return objects
        
    def _createItem(self, object, depth=0):
        """ Create item for an object.
        """
        row=self.rowCount()
        self.setRowCount(self.rowCount()+1)
        height=0
        i=0
        for column in self._columns:
            property=self.dataAccessor().property(object,column)
            if property!=None:
                propertyWidget=PropertyView.propertyWidgetFromProperty(property)
                if propertyWidget.properyHeight()>height:
                    height=propertyWidget.properyHeight()
                text=str(propertyWidget.value())
            else:
                if Property.DEFAULT_HEIGHT>height:
                    height=Property.DEFAULT_HEIGHT
                text=""
            item=QTableWidgetItem(text)
            item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable)
            item.object=object
            self.setItem(row,i,item)
            i+=1
        self.verticalHeader().resizeSection(row,height)

    def itemSelectionChanged(self):
        """ Emits signal selected that the TabController can connect to.
        """
        logging.debug(__name__ + ": itemSelectionChanged")
        if not self._updatingFlag:
            self._selection = self.currentRow()
            if self.item(self.currentRow(),self._firstColumn)!=None:
                self.emit(SIGNAL("selected"), self.item(self.currentRow(),self._firstColumn).object)
            else:
                self.emit(SIGNAL("selected"), None)
        
    def select(self, object):
        """ Mark an object in the TableView as selected.
        """
        logging.debug(__name__ + ": select")
        items = []
        for i in range(self.rowCount()):
            if self.item(i,self._firstColumn).object == object:
                items += [(i)]
        if len(items) > 0:
            first=sorted(items)[0]
            self._selection = first
            self._updatingFlag +=1
            self.setCurrentCell(first,0)
            self._updatingFlag -=1

    def _selectedRow(self):
        if self._selection<self.rowCount():
            return self._selection
        elif self.rowCount()>0:
            return 0
        else:
            return None

    def restoreSelection(self):
        """ Restore selection.
        """
        logging.debug(__name__ + ": restoreSelection")
        select=self._selectedRow()
        if select != None:
            self._updatingFlag +=1
            self.setCurrentCell(select,0)
            self._updatingFlag -=1
    
    def selection(self):
        """ Return currently selected object.
        
        If selection unknown return first object.
        """
        #logging.debug(__name__ + ": selection")
        select=self._selectedRow()
        if select != None:
            return self.item(select,self._firstColumn).object
        else:
            return None

    def isBusy(self):
        return self._updatingFlag>0

    def mousePressEvent(self,event):
        QTableWidget.mousePressEvent(self,event)
        if event.button()==Qt.RightButton:
            self.emit(SIGNAL("mouseRightPressed"), event.globalPos())
