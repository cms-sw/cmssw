class EventFileAccessor(object):
    """ This class provides access to the underlying data model.
    """
    
    def open(self, filename):
        """ Open file and read first event.
        """
        raise NotImplementedError
    
    def close(self):
        """ Close file.
        """
        pass
    
    def goto(self, index):
        """ Go to event number index and read it.
        """
        raise NotImplementedError

    def eventNumber(self):
        """ Return the current event number.
        """
        raise NotImplementedError

    def numberOfEvents(self):
        """ Return the total number of events.
        """
        raise NotImplementedError

