import logging

from PyQt4.QtCore import SIGNAL,Qt
from PyQt4.QtGui import QWidget

from Vispa.Share.ObjectHolder import ObjectHolder

class AbstractView(ObjectHolder):
    """ Abstract class for views which show a list of objects using a data accessor.
    
    A view handles the selection of objects and allows to restore selection after refreshing.
    On selection of an object the signal "selected" shall be emitted.
    """
    
    LABEL = "&Abstract View"
    
    def __init__(self):
        ObjectHolder.__init__(self)
        
    def updateContent(self):
        """ Update content of view.
        
        Return True if successful.
        """
        return True
        
    def select(self, object):
        """ Select an object in the view.
        """
        pass

    def selection(self):
        """ Return the last selected object in the view.
        """
        return None

    def restoreSelection(self):
        """ Select the last selected object in the view.
        """
        self.select(self.selection())

    def cancel(self):
        """ Stop all operations in view.
        """
        pass

    def isBusy(self):
        """ Return is operations are ongoing in view.
        """
        return False
    
class NoneView(AbstractView, QWidget):
    
    LABEL = "&None View"
    
    def __init__(self, parent=None):
        QWidget.__init__(self)
        AbstractView.__init__(self)

    def mousePressEvent(self,event):
        QWidget.mousePressEvent(self,event)
        if event.button()==Qt.RightButton:
            self.emit(SIGNAL("mouseRightPressed"), event.globalPos())
