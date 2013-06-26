import logging

from PyQt4.QtGui import QToolButton
from PyQt4.QtCore import Qt,SIGNAL

from Vispa.Share.BasicDataAccessor import BasicDataAccessor
from Vispa.Views.TableView import TableView

class CodeTableView(TableView):
    """ Table view that lists python configuration code.
    """

    LABEL = "Python configuration code"

    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        TableView.__init__(self,parent)
        self.setColumns(["Buttons","code"])
        self.horizontalHeader().hide()
        self.setSorting(False)
        self._firstColumn=1

        self.connect(self, SIGNAL("itemClicked(QTableWidgetItem*)"), self.itemClicked)

    def _createItem(self, object, properties):
        """ Create item for an object.
        """
        TableView._createItem(self, object, properties)
        button=QToolButton()
        button.object=object
        if self.dataAccessor().label(object)=="Import":
            button.setText("...")
            self.connect(button, SIGNAL('clicked(bool)'), self.importButtonClicked)
        elif self.dataAccessor().label(object)=="Apply tool":
            button.setText("+")
            self.connect(button, SIGNAL('clicked(bool)'), self.applyButtonClicked)
        else:
            button.setText("X")
            self.connect(button, SIGNAL('clicked(bool)'), self.removeButtonClicked)
        self.setCellWidget(self.rowCount()-1, 0, button)

    def importButtonClicked(self,checked=False):
        self.emit(SIGNAL("importButtonClicked"),self.sender().object)
    
    def applyButtonClicked(self,checked=False):
        self.emit(SIGNAL("applyButtonClicked"),self.sender().object)
        
    def removeButtonClicked(self,checked=False):
        self.emit(SIGNAL("removeButtonClicked"),self.sender().object)
        
    def updateContent(self):
        result=TableView.updateContent(self)
        self.horizontalHeader().resizeSection(0,self.cellWidget(0,0).sizeHint().width())
        return result
    
    def keyPressEvent(self,event):
        TableView.keyPressEvent(self,event)
        if event.key() in [Qt.Key_Backspace,Qt.Key_Delete]:    
            self.emit(SIGNAL("removeButtonClicked"),self.selection())

    def itemClicked(self,item):
        logging.debug(__name__ + ": itemClicked")
        if not self._updatingFlag:
            self.setCurrentCell(self.currentRow(),0)
