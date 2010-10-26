
class UndoEvent(object):
    LABEL = ""
    
    def __init__(self):
        pass
    
    def undo(self):
        """ Undos this event.
        """
        raise NotImplementedError
    
    def redo(self):
        """ Repeats this event.
        """
        raise NotImplementedError
    
    def combine(self, otherUndoEvent):
        """ Combines this event with another event of the same kind.
        
        If the combination was successfull True or otherwise False will be returned.
        """
        return False
    
    def description(self):
        """ Returns a string with more detailed explanation of what this event does
        
        E.g. what exactly was dragged, moved or which values changed.
        """
        return ""
    
    def setLastSavedState(self, flag):
        """ Sets the last saved state flag.
        
        If the flag is True, this UndoEvent represents the first action 
        after saving the file for the last time.
        """
        self._lastSavedStateFlag = flag
        
    def isLastSavedState(self):
        """ Returns the last saved state flag, see setLastSavedState().
        """
        if hasattr(self, "_lastSavedStateFlag"):
            return self._lastSavedStateFlag
        return False
    
    def dump(self, prefix="undo"):
        print prefix, ": ", self.LABEL, self.isLastSavedState()
    
    
class MultiUndoEvent(UndoEvent):
    """ This UndoEvent holds a list of UndoEvents whose redo() and undo() are invoked at the same time
    when the corresponding function of this event is invoked.
    """
    
    LABEL = "Multiple actions"
    
    def __init__(self, listOfUndoEvents, label=None):
        UndoEvent.__init__(self)
        self._undoEvents = listOfUndoEvents
        if label:
            self.LABEL = label
        if len(self._undoEvents) == 1:
            self.LABEL = self._undoEvents[0].LABEL
            
        labels = []
        for event in self._undoEvents:
            if not event.LABEL in labels:
                labels.append(event.LABEL)
        if len(labels) > 0:
            self.LABEL += " (%s)" % ", ".join(labels)
    
    def undo(self):
        for event in self._undoEvents:
            event.undo()
    
    def redo(self):
        # undo event list comes sorted so that it will work for undo
        # if events depend on each other its important to reverse order for redo()
        for event in reversed(self._undoEvents):
            event.redo()
            
    def dump(self, prefix="undo"):
        UndoEvent.dump(self, prefix)
        for event in self._undoEvents:
            event.dump("  " + prefix)
            