import logging

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Vispa.Main.Application import Application
from Vispa.Share.BasicDataAccessor import BasicDataAccessor
from Vispa.Views.AbstractView import AbstractView
from Vispa.Views.PropertyView import PropertyView,Property
from Vispa.Share.ThreadChain import ThreadChain

class TableWidgetItem(QTableWidgetItem):
    def __lt__(self,other):
        return str(self.text()).lower()<str(other.text()).lower()

class TableView(AbstractView, QTableWidget):
    """ The TableView widget fills itself using a DataAccessor.
    """
    
    LABEL = "&Table View"
    UPDATE_EVERY = 20
    
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        AbstractView.__init__(self)
        QTableWidget.__init__(self, parent)

        self._operationId = 0
        self._selection = (None,None)
        self._updatingFlag = 0
        self._columns=[]
        self._sortingFlag=False
        self._filteredColumns=[]
        self._firstColumn=0
        self._updateCounter=0
        self._autosizeColumns=True

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
        objects=self.allDataObjectChildren()
        properties=[]
        for object in objects:
            thread = ThreadChain(self.dataAccessor().properties, object)
            while thread.isRunning():
                if not Application.NO_PROCESS_EVENTS:
                    QCoreApplication.instance().processEvents()
            if operationId != self._operationId:
                self._updatingFlag-=1
                return False
            properties+=[thread.returnValue()]
        if self._filteredColumns!=[]:
            self._columns=self._filteredColumns
        else:
            self._columns=[]
            ranking={}
            for ps in properties:
              for property in ps:
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
        i=0
        for object in objects:
            # Process application event loop in order to accept user input during time consuming drawing operation
            self._updateCounter+=1
            if self._updateCounter>=self.UPDATE_EVERY:
                self._updateCounter=0
                if not Application.NO_PROCESS_EVENTS:
                    QCoreApplication.instance().processEvents()
            # Abort drawing if operationId out of date
            if operationId != self._operationId:
                break
            self._createItem(object,properties[i])
            i+=1
        if self._autosizeColumns:
            for i in range(len(self._columns)):
                self.resizeColumnToContents(i)
        self.setSortingEnabled(self._sortingFlag)
        self._updatingFlag-=1
        return self._operationId==operationId

    def _createItem(self, object, properties):
        """ Create item for an object.
        """
        row=self.rowCount()
        self.setRowCount(self.rowCount()+1)
        height=Property.DEFAULT_HEIGHT
        firstColumnDone=False
        for property in properties:
            if property!=None and property[1] in self._columns:
                i=self._columns.index(property[1])
                if property[0] in ["MultilineString","Double"]: 
                    propertyWidget=PropertyView.propertyWidgetFromProperty(property)
                    if propertyWidget.properyHeight()>height:
                        height=propertyWidget.properyHeight()
                    text=str(propertyWidget.value())
                else:
                    text=str(property[2])
                item=TableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable)
                item.object=object
                self.setItem(row,i,item)
                if i==self._firstColumn:
                    firstColumnDone=True
        if not firstColumnDone:
            item=QTableWidgetItem("")
            item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable)
            item.object=object
            self.setItem(row,self._firstColumn,item)
        self.verticalHeader().resizeSection(row,height)

    def itemSelectionChanged(self):
        """ Emits signal selected that the TabController can connect to.
        """
        logging.debug(__name__ + ": itemSelectionChanged")
        if not self._updatingFlag:
            self._selection = (self.currentRow(),self.item(self.currentRow(),self._firstColumn).text())
            if self.item(self.currentRow(),self._firstColumn)!=None:
                self.emit(SIGNAL("selected"), self.item(self.currentRow(),self._firstColumn).object)
            else:
                self.emit(SIGNAL("selected"), None)
        
    def select(self, object):
        """ Mark an object in the TableView as selected.
        Remember position an name of object in order to restore the selection.
        """
        logging.debug(__name__ + ": select")
        items = []
        for i in range(self.rowCount()):
            if self.item(i,self._firstColumn).object == object:
                items += [(i,self.item(i,self._firstColumn))]
        if len(items) > 0:
            index = items[0][0]
            item = items[0][1]
            self._selection = (index,item.text())
            self._updatingFlag +=1
            self.setCurrentItem(item)
            self._updatingFlag -=1

    def _selectedRow(self):
        """ Return the row containing the selected object.
        First search for object by name. If it is not found use position.
        """
        for i in range(self.rowCount()):
            if self.item(i,self._firstColumn).text() == self._selection[1]:
                return i
        if self._selection[0]<self.rowCount():
            return self._selection[0]
        if self.rowCount()>0:
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
