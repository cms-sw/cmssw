import logging

from PyQt4.QtGui import QColor

from Vispa.Share.BasicDataAccessor import BasicDataAccessor
from Vispa.Share.RelativeDataAccessor import RelativeDataAccessor
from Vispa.Share.ParticleDataAccessor import ParticleDataAccessor
from Vispa.Plugins.EventBrowser.EventFileAccessor import EventFileAccessor

class TestDataAccessor(BasicDataAccessor,RelativeDataAccessor,ParticleDataAccessor,EventFileAccessor):
    def __init__(self):
        self._eventNumber=1
    def children(self, parent):
        if parent=="container"+str(self._eventNumber):
            return ["particle1","subcontainer"]
        elif parent=="subcontainer":
            return ["particle2","particle3","particle4","particle5","particle6","particle7"]
        else:
            return []
        
    def isContainer(self, object):
        return object.startswith("container") or object.endswith("container")
                
    def motherRelations(self, object):
        if object=="particle4":
            return ["particle3"]
        if object=="particle5":
            return ["particle4"]
        if object=="particle6":
            return ["particle4"]
        else:
            return []

    def daughterRelations(self, object):
        if object=="particle3":
            return ["particle4"]
        if object=="particle4":
            return ["particle5","particle6"]
        else:
            return []

    def label(self, object):
        if object==None:
            return "Event"
        else:
            return object

    def properties(self, object):
        """ Make list of all properties """
        properties = []
        properties += [("Category", "Object info", "")]
        properties += [("String", "Label", object, None, False, True)]
        return properties

    def topLevelObjects(self):
        """ return top level objects from file, e.g. the event.
        """
        return ["container"+str(self._eventNumber),"anotherTopLevelObject"]

    def first(self):
        """ Go to first event and read it.
        """
        if self._eventNumber>1:
            self._eventNumber=1
            return True
        return False

    def previous(self):
        """ Go to previous event and read it.
        """
        self._eventNumber-=1
        if self._eventNumber<1:
            self._eventNumber=1
            return False
        return True

    def next(self):
        """ Go to next event and read it.
        """
        self._eventNumber+=1
        if self._eventNumber>3:
            self._eventNumber=3
            return False
        return True

    def last(self):
        """ Go to last event and read it.
        """
        if self._eventNumber<3:
            self._eventNumber=3
            return True
        return False

    def goto(self,index):
        """ Go to event number index and read it.
        """
        if index<1 or index>3:
            return False
        self._eventNumber=index
        return True

    def eventNumber(self):
        """ Return the current event number.
        """
        return self._eventNumber

    def numberOfEvents(self):
        """ Return the total number of events.
        """
        return 3

    def id(self,object):
        return object
    
    def isQuark(self,object):
        return False
    
    def isLepton(self,object):
        return False
    
    def isGluon(self,object):
        return False
    
    def isBoson(self,object):
        return False
    
    def particleId(self,object):
        return 13

    def addProperty(self, object, name, value, type):
        return True
    
    def removeProperty(self, object, name):
        return True

    def color(self, object):
        return QColor(176, 179, 177)

    def lineStyle(self, object):
        return self.LINE_STYLE_SOLID
    
    def linkMother(self, object, mother):
        return True
        
    def linkDaughter(self, object, daughter):
        return True
        
