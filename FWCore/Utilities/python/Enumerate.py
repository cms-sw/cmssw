#! /usr/bin/env python

##  Note: Please do not use or modify any data or functions with a
##  leading underscore.  If you "mess" with the internal structure,
##  the classes may not function as intended.

class Enumerate (object):
    """Similar to C++'s 'enum', but with a few extra toys.  Takes a
    string with spaces in between the different 'enum' names (keys).
    If 'asInt' is true, then values will be integers (useful for array
    indicies).  Once created, the enum values can not be changed."""

    def __init__(self, names, prefix = '', asInt = False, intOffset = 0):
        biggest = smallest = ""
        self._keys = []
        self._valueDict = {}
        for count, name in enumerate (names.split()) :
            # make sure we don't already have this key
            if self.isValidKey (name):
                raise RuntimeError, \
                      "You can not duplicate Enum Names '%s'" % name
            # set the value using the base class
            key = "%s_%s" % (prefix, name)
            if asInt:
                key = count + intOffset
            object.__setattr__ (self, name, key)
            self._valueDict[key] = name
            self._keys.append (name)


    def isValidValue (self, value):
        """ Returns true if this value is a valid enum value"""
        return self._valueDict.has_key (value)


    def isValidKey (self, key):
        """ Returns true if this value is a valid enum key"""
        return self.__dict__.has_key (key)


    def valueToKey (self, value):
        """ Returns the key (if it exists) for a given enum value"""
        return self._valueDict.get (value, None)


    def keys (self):
        """ Returns copy of valid keys """
        # since this is a list, return a copy of it instead of the
        # list itself
        return self._keys [:]


    def __setattr__ (self, name, value):
        """Lets me set internal values, but throws an error if any of
        the enum values are changed"""
        if not name.startswith ("_"):
            # Once it's set, you can't change the values
            raise RuntimeError, "You can not modify Enum values."
        else:
            object.__setattr__ (self, name, value)


    def __call__ (self, key):
        return self.__dict__.get (key, None)

