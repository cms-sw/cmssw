from PyQt4.QtCore import *

import logging 

class VispaWidgetOwner(object):
    """ Interface for classes containing VispaWidgets
    
    Only makes sense if implementing class also inherits QWidget or class inheriting QWidget.
    """
    
    def selectedWidgets(self):
        """ Returns a list of all selected widgets.
        """
        return [child for child in self.children() if hasattr(child, "isSelected") and child.isSelected()]
    
    def widgetSelected(self, widget):
        """ Forward selection information to super class if it is a VispaWidgetOwner.
        """
        #logging.debug(self.__class__.__name__ +": widgetSelected()")
        if isinstance(self, QObject):
            self.deselectAllWidgets(widget)
            if isinstance(self.parent(), VispaWidgetOwner):
                self.parent().widgetSelected(widget)
            
    def widgetMoved(self, widget):
        """ Tell parent widget has moved.
        
        Only informs parent if it is a VispaWidgetOwner, too.
        """
        if isinstance(self.parent(), VispaWidgetOwner):
            self.parent().widgetMoved(widget)
    
    def widgetAboutToDelete(self, widget):
        """ Tells parent widget is about to delete.
        
        This function is called from the delete() function of VispaWidget.
        """
        # Does VispaWidgetOwner really need this?
        # Enough if ConnectableWidgetOwner has this function?
        if isinstance(self.parent(), VispaWidgetOwner):
            self.parent().widgetAboutToDelete(widget)
    
    def deselectAllWidgets(self, exception=None):
        """ Deselects all widgets except the widget given as exception.
        """
        #logging.debug(self.__class__.__name__ +": deselectAllWidgets()")
        for child in self.children():
            if child != exception and hasattr(child, 'select'):
                child.select(False)
            if isinstance(child, VispaWidgetOwner):
                child.deselectAllWidgets(exception)
        self.update()
        
    def mousePressEvent(self, event):
        """ Calls deselectAllWidgets.
        """
        self.deselectAllWidgets()