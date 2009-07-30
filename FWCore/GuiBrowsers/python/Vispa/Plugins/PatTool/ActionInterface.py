class ActionInterface(object):
    def dumpPython(self):
        """ Return the python code to perform the action.
        """
        raise NotImplementedError
    def label(self):
        """ Return the label of the action.
        """
        raise NotImplementedError
    def description(self):
        """ Return a string with a detailed description of the action.
        """
        return ""
    def parameters(self):
        """ Return the list of the parameters of an action.
        
        Each parameters is represented by a tuple containing its
        type, name, value and description.
        The type determines how the parameter is represented in the GUI.
        Possible types are: 'Category','String','Text','File','FileVector','Boolean','Integer','Float'.
        """
        raise NotImplementedError
    def setParameter(self, name, value):
        """ Change the parameter 'name' to a new value.
        """
        raise NotImplementedError
