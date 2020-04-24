class BasicDataAccessor(object):
    """ This class provides access to the underlying data model.
    """
    
    def children(self, object):
        """ Return the children of a container object.
        """
        raise NotImplementedError
    
    def isContainer(self, object):
        """ Return if the object is a container object.
        """
        raise NotImplementedError

    def label(self, object):
        """ Return a string that is used as caption of an object.
        """
        raise NotImplementedError

    def properties(self, object):
        """ Return the list of the properties of an object.
        
        Each property is represented by a tuple containing its
        type, name, value, description, readonly(True/False), deletable(True/False).
        Possible types are: 'Category','String','MultilineString','File','FileVector','Boolean','Integer','Double'.
        """
        raise NotImplementedError
    
    def setProperty(self, object, name, value, categoryName):
        """ Change the property 'name' of an object to a new value.
        """
        raise NotImplementedError

    def addProperty(self, object, name, value, type):
        """ Add the property 'name' to an object.
        """
        raise NotImplementedError

    def removeProperty(self, object, name):
        """ Remove the property 'name' from an object.
        """
        raise NotImplementedError

    def property(self, object, name):
        """ Returns property with given name.
        """
        propertiesDict = {}
        for p in self.properties(object):
            propertiesDict[p[1]] = p
        if name in propertiesDict.keys():
            return propertiesDict[name]
        else:
            return None

    def propertyValue(self, object, name):
        """ Returns value of property with given name.
        """
        property=self.property(object,name)
        if property!=None:
            return property[2]
        else:
            return None

    def allChildren(self, object):
        """ Collect all children of children of an object.
        """
        children = []
        for child in self.children(object):
            children += [child]+self.allChildren(child)
        return children

    def topLevelObjects(self):
        """ Return top level objects, e.g. the event.
        """
        raise NotImplementedError

class BasicDataAccessorInterface(object):
    """ This class gives a comfortable Interface to objects accessible via an accessor.
    
    Given the object and the accessor all properties and attributes of the object and
    the accessor are accessible via __getattr__. A script in which all attributes
    of the objects can be accessed can be run. 
    """
    def __init__(self, object, accessor, throwAttributeErrors=True):
        self._object = object
        self._accessor = accessor
        self._throwAttributeErrors=throwAttributeErrors
        
    def __getattr__(self, attr):
        if attr in [p[1] for p in self._accessor.properties(self._object)]:
            return self._accessor.propertyValue(self._object, attr)
        elif hasattr(self._object, attr):
            return getattr(self._object, attr)
        elif hasattr(self._accessor, attr):
            return getattr(self._accessor, attr)(self._object)
        else:
            if self._throwAttributeErrors:
                raise AttributeError("object has no property '" + attr + "'")
            else:
                return "???"

    def runScript(self, script):
        object = self
        exec("result=" + str(script))
        return result
