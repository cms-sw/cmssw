from PyQt4.QtCore import QObject, QRect, QPoint, Qt
from PyQt4.QtGui import QApplication

import logging 

class VispaWidgetOwner(object):
    """ Interface for classes containing VispaWidgets
    
    Only makes sense if implementing class also inherits QWidget or class inheriting QWidget.
    """
    
    def enableMultiSelect(self, multiSelect=True):
        self._multiSelectEnabledFlag = multiSelect
        
    def multiSelectEnabled(self):
        return hasattr(self, "_multiSelectEnabledFlag") and self._multiSelectEnabledFlag
    
    def selectedWidgets(self):
        """ Returns a list of all selected widgets.
        """
        if hasattr(self, "_selectedWidgets"):
            return self._selectedWidgets
        return [child for child in self.children() if hasattr(child, "isSelected") and child.isSelected()]
    
    def widgetSelected(self, widget, multiSelect=False):
        """ Forward selection information to super class if it is a VispaWidgetOwner.
        """
        logging.debug(self.__class__.__name__ +": widgetSelected()")

        if isinstance(self, QObject):
            if not hasattr(self, "_selectedWidgets"):
                self._selectedWidgets = []
                
            if  not multiSelect or not self.multiSelectEnabled():
                self.deselectAllWidgets(widget)
                self._selectedWidgets = []
                
            if widget.parent() == self and not widget in self._selectedWidgets:
                self._selectedWidgets.append(widget)
                
            for widget in [child for child in self._selectedWidgets if hasattr(child, "isSelected") and not child.isSelected()]:
                self._selectedWidgets.remove(widget)
                
            if isinstance(self.parent(), VispaWidgetOwner):
                self.parent().widgetSelected(widget)
            
    def widgetDoubleClicked(self, widget):
        """ Forward selection information to super class if it is a VispaWidgetOwner.
        """
        #logging.debug(self.__class__.__name__ +": widgetDoubleClicked()")
        if isinstance(self, QObject):
            if isinstance(self.parent(), VispaWidgetOwner):
                self.parent().widgetDoubleClicked(widget)
                
    def initWidgetMovement(self, widget):
        self._lastMovedWidgets = []
        if self.multiSelectEnabled():
            pos = widget.pos()
            for child in self.children():
                if child != widget and hasattr(child, "isSelected") and child.isSelected():
                    child.setDragReferencePoint(pos - child.pos())
            
    def widgetDragged(self, widget):
        """ Tell parent widget has moved.
        
        Only informs parent if it is a VispaWidgetOwner, too.
        """
        if isinstance(self.parent(), VispaWidgetOwner):
            self.parent().widgetDragged(widget)

        if hasattr(self, "_lastMovedWidgets"):
            self._lastMovedWidgets.append(widget)
            
        if self.multiSelectEnabled():
            for child in self.children():
                if hasattr(child, "dragReferencePoint") and child != widget and hasattr(child, "isSelected") and child.isSelected():
                    if hasattr(child, "setPreviousDragPosition"):
                        child.setPreviousDragPosition(child.pos())
                    child.move(widget.pos() - child.dragReferencePoint())
                    self._lastMovedWidgets.append(child)

# apparently unused feature (2010-07-02), remove if really unnecessary
# also see self._lastMovedWidget definition above
    def lastMovedWidgets(self):
        if hasattr(self, "_lastMovedWidgets"):
            return self._lastMovedWidgets
        return None

    def widgetAboutToDelete(self, widget):
        """ This function is called from the delete() function of VispaWidget.
        """
        pass
        
    def keyPressEvent(self, event):
        """ Calls delete() method of selected child widgets if multi-select is activated.
        """
        if self.multiSelectEnabled() and ( event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete ):
            selection = self.selectedWidgets()[:]
            for widget in selection:
                widget.delete()
    
    def deselectAllWidgets(self, exception=None):
        """ Deselects all widgets except the widget given as exception.
        """
        #logging.debug(self.__class__.__name__ +": deselectAllWidgets()")
        self._selectedWidgets = []
        for child in self.children():
            if child != exception:
                if hasattr(child, 'select'):
                    child.select(False)
            else:
                self._selectedWidgets.append(child)
                
            if isinstance(child, VispaWidgetOwner):
                child.deselectAllWidgets(exception)
        self.update()
        
    def mousePressEvent(self, event):
        """ Calls deselectAllWidgets.
        """
        multiSelectEnabled = self.multiSelectEnabled()
        if event.modifiers() != Qt.ControlModifier:
            self.deselectAllWidgets()
        if multiSelectEnabled:
            self._selectionRectStartPos = QPoint(event.pos())
            self._selectionRect = None

    def mouseMoveEvent(self, event):
        if self.multiSelectEnabled() and self._selectionRectStartPos and (event.pos() - self._selectionRectStartPos).manhattanLength() >= QApplication.startDragDistance():
            eventX = event.pos().x()
            eventY = event.pos().y()
            startX = self._selectionRectStartPos.x()
            startY = self._selectionRectStartPos.y()
            oldRect = self._selectionRect
            self._selectionRect = QRect(min(startX, eventX), min(startY, eventY), abs(eventX - startX), abs(eventY - startY))
            if oldRect:
                self.update(self._selectionRect.united(oldRect).adjusted(-5, -5, 5, 5))
            else:
                self.update(self._selectionRect)
                
            # dynamically update selection statur
            # currently bad performance (2010-07-07)
            # TODO: improve selection mechanism
#            for child in self.children():
#                if hasattr(child, "select") and hasattr(child, "isSelected"):
#                    child.select(self._selectionRect.contains(child.geometry()), True)    # select, mulitSelect 
                
    def mouseReleaseEvent(self, event):
        if hasattr(self, "_selectionRect") and  self._selectionRect and self.multiSelectEnabled():
            for child in self.children():
                if hasattr(child, "select") and hasattr(child, "isSelected") and self._selectionRect.contains(child.geometry()) and not child.isSelected():
                    child.select(True, True)    # select, mulitSelect 
            self.update(self._selectionRect.adjusted(-5, -5, 5, 5))
        self._selectionRect = None
        self._selectionRectStartPos = None
        