import logging

class ObjectHolder(object):
    """ Abstract class for holders of objects which are accessed via a data accessor.
    
    Objects can be filtered using a filter function. 
    """
    def __init__(self):
        logging.debug(__name__ + ": __init__")
        self._dataAccessor = None
        self._dataObjects = []
        self._filter = self._noFilter
        
    def setDataAccessor(self, accessor):
        """ Sets the DataAccessor from which the nodes are created.
        
        You need to call updateContent() in order to make the changes visible.   
        """
        self._dataAccessor = accessor
    
    def dataAccessor(self):
        return self._dataAccessor
    
    def setDataObjects(self, objects):
        """ Sets the objects that shall be shown.
        
        You need to call updateContent() in order to make the changes visible.   
        """
        self._dataObjects = objects
        
    def dataObjects(self):
        return self._dataObjects

    def setFilter(self, filter):
        """ Set the filter function used in the view.
        """
        self._filter = filter
        
    def _noFilter(self, objects):
        """ The default filter function for objects.
        """
        return objects
