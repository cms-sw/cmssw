class BasicDataAccessor(object):
    """ This class provides access to the underlying data model.
    """
    
    def children(self, parent):
        """ Return the children of a container object.
        """
        raise NotImplementedError

    def label(self, object):
        """ Return a string that is used as caption of an object.
        """
        raise NotImplementedError

    def properties(self, object):
        """ Return the list of the properties of an object.
        
        Each property is represented by a tuple containing its
        type, name and value.
        """
        raise NotImplementedError
    
    def propertyValue(self, object, name):
        """ Returns value of property with given name.
        """
        raise NotImplementedError
    
    def setProperty(self, object, name, value):
        """ Change the property 'name' of an object to a new value.
        """
        raise NotImplementedError

    def allChildren(self,object):
        children=list(self.children(object))
        for child in self.children(object):
            children+=list(self.allChildren(child)) 
        return tuple(children)

class BasicDataAccessorInterface(object):
    def __init__(self,object,accessor):
        self._object=object
        self._accessor=accessor
        
    def __getattr__(self,attr):
        if hasattr(self._accessor,attr):
            return getattr(self._accessor,attr)(self._object)
        elif attr in [p[1] for p in self._accessor.properties(self._object)]:
            return self._accessor.propertyValue(self._object,attr)
        else:
            raise AttributeError("object has no property '"+attr+"'")

    def applyScript(self,script):
        object=self
        exec "result="+str(script)
        return result
