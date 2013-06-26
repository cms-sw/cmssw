import logging

from Vispa.Views.TableView import TableView

class BranchTableView(TableView):
    """ Table view that lists python configuration code.
    """

    LABEL = "Branches"

    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        TableView.__init__(self,parent)
        self.setColumns(["Type","Label","Product","Process"])
        self._firstColumn=0
        self._autosizeColumns=False
        self._sortingFlag=True

    def allDataObjectChildren(self):
        return self.dataObjects()
    
    def selection(self):
        return self.dataAccessor().read(TableView.selection(self))
