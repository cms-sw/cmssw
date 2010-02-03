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
        return [child for child in self.children() if hasattr(child, "isSelected") and child.isSelected()]
    
    def widgetSelected(self, widget, multiSelect=False):
        """ Forward selection information to super class if it is a VispaWidgetOwner.
        """
        #logging.debug(self.__class__.__name__ +": widgetSelected()")

        if isinstance(self, QObject):
            if not self.multiSelectEnabled() or not multiSelect:
                self.deselectAllWidgets(widget)
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
        self._lastMovedWidget = None
        if self.multiSelectEnabled():
            pos = widget.pos()
            for child in self.children():
                if child != widget and hasattr(child, "isSelected") and child.isSelected():
                    child.setDragReferencePoint(pos - child.pos())
            
    def widgetMoved(self, widget):
        """ Tell parent widget has moved.
        
        Only informs parent if it is a VispaWidgetOwner, too.
        """
        if isinstance(self.parent(), VispaWidgetOwner):
            self.parent().widgetMoved(widget)
        
        self._lastMovedWidget = widget
            
        if self.multiSelectEnabled():
            for child in self.children():
                if hasattr(child, "dragReferencePoint") and child != widget and hasattr(child, "isSelected") and child.isSelected():
                    child.move(widget.pos() - child.dragReferencePoint())

    def lastMovedWidget(self):
        if hasattr(self, "_lastMovedWidget"):
            return self._lastMovedWidget
        return None

    def widgetAboutToDelete(self, widget):
        """ Tells parent widget is about to delete.
        
        This function is called from the delete() function of VispaWidget.
        """
        if isinstance(self.parent(), VispaWidgetOwner):
            self.parent().widgetAboutToDelete(widget)
        
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
        for child in self.children():
            if child != exception and hasattr(child, 'select'):
                child.select(False)
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
                
    def mouseReleaseEvent(self, event):
        if hasattr(self, "_selectionRect") and  self._selectionRect and self.multiSelectEnabled():
            for child in self.children():
                if hasattr(child, "select") and hasattr(child, "isSelected") and self._selectionRect.contains(child.geometry()) and not child.isSelected():
                    child.select(True, True)    # select, mulitSelect 
            self.update(self._selectionRect.adjusted(-5, -5, 5, 5))
        self._selectionRect = None
        self._selectionRectStartPos = None
        