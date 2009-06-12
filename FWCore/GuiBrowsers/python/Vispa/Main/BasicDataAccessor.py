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

    def allChildren(self, object):
        """ Collect all children of children of an object.
        """
        children = []
        for child in self.children(object):
            children += [child]+list(self.allChildren(child))
        return tuple(children)

class BasicDataAccessorInterface(object):
    """ This class gives a comfortable Interface to objects accessible via an accessor.
    
    Given the object and the accessor all properties and attributes of the object and
    the accessor are accessible via __getattr__. A script in which all attributes
    of the objects can be accessed can be run. 
    """
    def __init__(self, object, accessor):
        self._object = object
        self._accessor = accessor
        
    def __getattr__(self, attr):
        if attr in [p[1] for p in self._accessor.properties(self._object)]:
            return self._accessor.propertyValue(self._object, attr)
        elif hasattr(self._object, attr):
            return getattr(self._object, attr)
        elif hasattr(self._accessor, attr):
            return getattr(self._accessor, attr)(self._object)
        else:
            raise AttributeError("object has no property '" + attr + "'")

    def runScript(self, script):
        object = self
        exec "result=" + str(script)
        return result
