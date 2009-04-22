
import logging

from PyQt4.QtGui import *

from ConnectableWidget import *
from BasicDataAccessor import *
from RelativeDataAccessor import *
from ParticleDataAccessor import *
from Workspace import *

import Resources

#class is too full, but it works?
class NoneCenterView(Workspace):
    """
    Disables the centerView - However, widgets have to be vreated, to use the right panel!
    """
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        Workspace.__init__(self, parent)

        self._accessor = None
        self._dataObjects = []
        self._operationId = 0
        self._boxContentScript = ""
        self._sortBeforeArranging = True
        self._useLineDecayTree = False
       
    def setDataAccessor(self, accessor):
        """ Sets the DataAccessor from which the boxes are created.
        
        You need to call updateContent() in order to make the changes visible.   
        """
        if not isinstance(accessor, BasicDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type BasicDataAccessor.")
        if not isinstance(accessor, RelativeDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type RelativeDataAccessor.")
        self._accessor = accessor
    
    def accessor(self):
        return self._accessor
    
    def setDataObjects(self, objects):
        """ Sets the selected object from which the boxes are created
        
        You need to call updateContent() in order to make the changes visible   
        """
        self._dataObjects = objects
        
    def dataObjects(self):
        return self._dataObjects

    def clear(self):
        """ Deletes all boxes in the BoxDecayTree
        """
        logging.debug(__name__ + ": clear")
        # Abort currently ongoing drawing operations
        self._operationId += 1
        Workspace.clear(self)

    def updateContent(self):
        """ Clear the BoxDecayTree and refill it.
        """
        logging.debug(__name__ + ": updateContent")
        self.clear()
        operationId = self._operationId
        if self._accessor:
            objects = self._dataObjects
            i = 0
            for object in objects:
                if operationId != self._operationId:
                    break
                self.createBoxRecursive(operationId, object, self, str(i))
                i += 1
        #self.show()

    def createBox(self, widgetParent, container, title, text):
        """ Create a WidgetContainer or ConnectableWidget and set its properties.
        """
        widget = ConnectableWidget(widgetParent)
        return widget


    def createBoxRecursive(self, operationId, object=None, widgetParent=None, id="0"):
        """ Creates a box from an object.
        
        All children of this object are created recursively.
        """
        # Abort drawing if operationId out of date
        if operationId != self._operationId:
            return None

        # Process application event loop in order to accept user input during time consuming drawing operation
        qApp.processEvents()
        
        # create box
        text = ""
        widget = self.createBox(widgetParent, len(self._accessor.children(object)) > 0, self._accessor.label(object), text)
        self.addWidget(widget, object, id)



        
    
