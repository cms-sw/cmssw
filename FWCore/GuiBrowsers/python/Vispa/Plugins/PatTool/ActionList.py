from ActionInterface import ActionInterface

class ActionList(object):
    def __init__(self):
        self._actionList=[]
        self._actionListHistory=[]
    def dumpPython(self):
        """ Return the python code to perform all actions in the list.
        """
        raise NotImplementedError
    def append(self,action):
        """ Append action to list and record old list in history.
        """
        if not isinstance(action,ActionInterface):
            raise TypeError("Action list can only hold objects of type ActionInterface")
        raise NotImplementedError
    def insert(self,action,index=0):
        """ Insert action into list and record old list in history.
        """
        if not isinstance(action,ActionInterface):
            raise TypeError("Action list can only hold objects of type ActionInterface")
        raise NotImplementedError
    def remove(self,action):
        """ Remove action from list and record old list in history.
        """
        raise NotImplementedError
    def edit(self,action):
        """ Record copy of action in history.
        """
        raise NotImplementedError
    def unDo(self):
        """ Undo one step in history.
        """
        raise NotImplementedError
    def reDo(self):
        """ Redo one step in history.
        """
        raise NotImplementedError
