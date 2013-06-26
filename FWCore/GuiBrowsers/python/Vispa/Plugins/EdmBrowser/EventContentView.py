import logging

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Vispa.Views.AbstractView import AbstractView

class EventContentView(QTableWidget, AbstractView):
    """ Holds a table to compare event contents.
    """
    def __init__(self, parent=None, name=None):
        """ Constructor """
        logging.debug(self.__class__.__name__ + ": __init__()")
        AbstractView.__init__(self)
        QTableWidget.__init__(self, parent)
    
        self._itemDict = {}
       
        self.setSelectionMode(QTableWidget.SingleSelection)
        self.clear() # Sets headers

    def clear(self):
        self._itemDict = {}
        QTableWidget.clear(self)
        self.setColumnCount(1)
        self.setHorizontalHeaderLabels(['no file selected'])
        self.horizontalHeaderItem(0).setTextAlignment(Qt.AlignLeft)
        self.verticalHeader().hide()
        self._rows = 0
        self.setRowCount(self._rows)

    def _addRow(self, names, items):
        """ Adds a row to the EventContentView.
        """
        self._rows += 1
        i=0
        for name in names:
            item = LabelItem(name)
            self.setItem(self._rows - 1, i, item)
            i+=1
        for entry, color in items:
            item = LabelItem(entry, color)
            self.setItem(self._rows - 1, i, item)
            if i == 4:
                self._itemDict[str(self._rows)] = item
                item.object = name
            i += 1
    
    def updateContent(self):
        """ Fill the EventContentView using the ContentsDict.
        """
        logging.debug('EventContentView: updateContent()')
        self._eventContentsList = self.dataAccessor().eventContentsList()
        self.clear()
        self.setSortingEnabled(False)
        names = ["Type","Label","Product","Process"]
        allContent = []
        for name, content, relative, comment in self._eventContentsList:
            names += [name]
            for entry in content:
                if not self.dataAccessor().inContent(entry,allContent):
                    allContent += [entry]
        allContent.sort(lambda x, y: cmp(self.dataAccessor().label(x).lower(), self.dataAccessor().label(y).lower()))
        self.setColumnCount(len(names))
        self.setHorizontalHeaderLabels(names)
        for i in range(len(names)):
            self.horizontalHeaderItem(i).setTextAlignment(Qt.AlignLeft)
        self.setRowCount(len(allContent))
        for entry in allContent:
            items = []
            column_before=None
            for name, content, relative, comment in self._eventContentsList:
                this_column=self.dataAccessor().inContent(entry,content)
                if this_column:
                    text="Yes"
                else:
                    text="No"
                if "_".join(entry) in comment.keys():
                    text+=" ("+comment["_".join(entry)]+")"
                #if relative:
                color=Qt.white
                #else:
                #    color=Qt.gray
                input="Input: " in name
                output="Output: " in name
                if column_before!=None:
                    rel=0
                    if this_column and not column_before:
                        rel=1
                    if not this_column and column_before:
                        rel=-1
                    if input:
                        if rel>0:
                            color=Qt.red
                    elif output:
                        if rel>0:
                            color=Qt.green
                    else:
                        if rel<0:
                            color=Qt.red
                        if rel>0:
                            color=Qt.green
                if not input:
                    column_before=this_column
                items += [(text, color)]
            self._addRow(entry,items)
        self.sortItems(1)
        self.setSortingEnabled(True)
        return True

    def select(self, item):
        """ Mark an item in the TableView as selected.
        """
        if item != None:
            self.setCurrentItem(item)

    def itemByObject(self, object):
        """ Return an item in the TableView with a certain object.
        """
        items = []
        for positionName, item in self._itemDict.items():
            if item.object == self.dataAccessor().label(object):
                items += [(positionName, item)]
        if len(items) > 0:
            return sorted(items)[0][1]
        else:
            return None

class LabelItem(QTableWidgetItem):
    """ A QTableWidgetItem with a convenient constructor. 
    """
    def __init__(self, argument, color=Qt.white):
        tooltip = argument
        name = argument
        QTableWidgetItem.__init__(self, name)
        self.setToolTip(tooltip)
        self.setFlags(Qt.ItemIsEnabled)
        self.setBackgroundColor(color)
