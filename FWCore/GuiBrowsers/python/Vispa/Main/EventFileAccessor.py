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
        raise NotImplementedError
    
    def topLevelObjects(self):
        """ Return top level objects from file, e.g. the event.
        """
        raise NotImplementedError

    def first(self):
        """ Go to first event and read it.
        """
        raise NotImplementedError

    def previous(self):
        """ Go to previous event and read it.
        """
        raise NotImplementedError

    def next(self):
        """ Go to next event and read it.
        """
        raise NotImplementedError

    def last(self):
        """ Go to last event and read it.
        """
        raise NotImplementedError

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

