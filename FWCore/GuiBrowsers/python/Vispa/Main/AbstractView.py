import logging

from ObjectHolder import ObjectHolder

class AbstractView(ObjectHolder):
    """ Abstract class for views which show a list of objects using a data accessor.
    
    A view handles the selection of objects and allows to restore selection after refreshing.
    On selection of an object the signal "selected" shall be emitted.
    """
    def __init__(self):
        ObjectHolder.__init__(self)
        
    def updateContent(self):
        """ Update content of view.
        """
        pass
        
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
