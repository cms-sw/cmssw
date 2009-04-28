import logging

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Vispa.Main.AbstractTab import *
from Vispa.Main.BasicDataAccessor import *

class PropertyView(QTableWidget):
    """ Shows properties of an object in a QTableWidget using the DataAccessor.
    """
    def __init__(self, parent=None, name=None):
        """ Constructor """
        logging.debug(self.__class__.__name__ + ": __init__()")
        QTableWidget.__init__(self, parent)
        self._accessor = None
       
        self.updateIni = False
        self.setSortingEnabled(False)
        self.verticalHeader().hide()
        self.setSelectionMode(QTableWidget.NoSelection)
        self._rows = 0
        self.setColumnCount(2)
        self.clear()        # sets header

        self._accessor = None
        self._dataObject = None
        self._readOnly = False
        
        self.connect(self, SIGNAL("itemDoubleClicked(QTableWidgetItem *)"), self.itemDoubleClickedSlot)
        
    def clear(self):
        """ Clear the table and set the header label.
        """
        QTableWidget.clear(self)
        self._rows = 0
        self.setRowCount(self._rows)
        self.setHorizontalHeaderLabels(['Property', 'Value'])
    
    def _addEmptyRow(self):
        self._rows += 1
        self.setRowCount(self._rows)
        
    def updatePropertyHeight(self,property):
        #logging.debug(self.__class__.__name__ + ": updatePropertyHeight()")
        for i in range(self._rows):
            if self.cellWidget(i,1)==property:
                self.verticalHeader().resizeSection(i, property.properyHeight())
    
    def append(self, property):
        """ Adds a property to the PropertyView and returns it.
        """
        property.setPropertyView(self)
        property.setReadOnly(self._readOnly)
        self._addEmptyRow()
        self.setItem(self._rows - 1, 0, LabelItem(property))
        self.setCellWidget(self._rows - 1, 1, property)
        self.updatePropertyHeight(property)
        self.connect(property, SIGNAL('updatePropertyHeight'), self.updatePropertyHeight)
        return property
    
    def addCategory(self, name):
        """ Add a category row to the tabel which consits of two gray LabelItems.
        """
        self._addEmptyRow()
        self.setItem(self._rows - 1, 0, LabelItem(name, Qt.lightGray))
        self.setItem(self._rows - 1, 1, LabelItem("", Qt.lightGray))
        self.verticalHeader().resizeSection(self._rows - 1, Property.DEFAULT_HEIGHT)

    def setReadOnly(self, readOnly):
        """ Sets all properties in the PropertyView to read-only.
        
        After calling this function all properties that are added are set to read-only as well.
        """
        self._readOnly = readOnly
        for i in range(self._rows):
            property = self.cellWidget(i, 1)
            if property:
                property.setReadOnly(self._readOnly)
    
    def readOnly(self):
        return self._readOnly
    
    def resizeEvent(self, event):
        """ Resize columns when table size is changed.
        """
        if event != None:
            QTableWidget.resizeEvent(self, event)
        space = self.width() - 4
        if self.verticalScrollBar().isVisible():
            space -= self.verticalScrollBar().width()
        space -= self.columnWidth(0)
        self.setColumnWidth(1, space)
        if self.updateIni:
            self.writeIni()

    # TODO: this is still the Qt3 version
    def columnWidthChanged(self, col):
        """ Resize columns when column size is changed.
        """
        QTableWidget.columnWidthChanged(self, col)
        if col == 1:
            space = self.width() - 4
            if self.verticalScrollBar().isVisible():
                space -= self.verticalScrollBar().width()
            space -= self.columnWidth(1)
            if space != self.columnWidth(0):
                self.setColumnWidth(0, space)
        else:
            space = self.width() - 4
            if self.verticalScrollBar().isVisible():
                space -= self.verticalScrollBar().width()
            space -= self.columnWidth(0)
            if space != self.columnWidth(1):
                self.setColumnWidth(1, space)
                if self.updateIni:
                    self.writeIni()

    def setDataAccessor(self, accessor):
        """ Sets the DataAccessor from which the object properties are read.
        
        You need to call updateContent() in order to make the changes visible.
        """
        if not isinstance(accessor, BasicDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type BasicDataAccessor.")
        self._accessor = accessor
    
    def accessor(self):
        return self._accessor
    
    def setDataObject(self, object):
        """ Sets the object whose properties shall be shown.
        
        You need to call updateContent() in order to make the changes visible.   
        """
        self._dataObject = object
        
    def dataObject(self):
        return self._dataObject

    def updateContent(self):
        """ Fill the properties of an object in the PropertyView using the DataAccessor.
        """
        logging.debug('PropertyView: updateContent()')
        self.clear()
        if self._accessor == None or self._dataObject == None:
            return
        self._ignoreValueChangeFlag = True  # prevent infinite loop
        for property in self._accessor.properties(self._dataObject):
            if property[0] == "Category":
                self.addCategory(property[1])
            elif property[0] == "String":
                propertyObject = self.append(StringProperty(property[1], property[2]))
            elif property[0] == "Text":
                propertyObject = self.append(TextProperty(property[1], property[2]))
            elif property[0] == "File":
                propertyObject = self.append(FileProperty(property[1], property[2]))
            elif property[0] == "Boolean":
                propertyObject = self.append(BooleanProperty(property[1], property[2]))
                propertyObject.setChecked(property[2])      # strange, does not work in constructor
            elif property[0] == "Integer":
                propertyObject = self.append(IntegerProperty(property[1], property[2]))
                
            if len(property) > 3 and property[3] == "readonly":
                propertyObject.setReadOnly(True)
        self.resizeEvent(None)
        self._ignoreValueChangeFlag = False

    # TODO: not yet used
    def setIniProperties(self, porperties=None):
        if properties and "width" in properties.keys():
                width = properties["width"]
        else:
            width = WIDTH
        self.resize(width, 0)
        if properties and "columnwidth" in properties.keys():
            columnwidth = properties["columnwidth"]
        else:
            columnwidth = COLUMN_WIDTH
        self.setColumnWidth(0, columnwidth)

    # TODO: ini not yet used
    def getIniProperties(self):
        properties = {}
        properties["width"] = self.width()
        properties["columnwidth"] = self.columnWidth(0)
        return properties

    def valueChanged(self, property):
        """ This function is called when a property a changed.
        
        The DataAcessor is called to handle the property change.
        """
        if self._accessor and not self._ignoreValueChangeFlag:
            oldValue = self._accessor.propertyValue(self._dataObject, property.name())
            if self._accessor.setProperty(self._dataObject, property.name(), property.value()):
                property.setHighlighted(False)
                if oldValue != self._accessor.propertyValue(self._dataObject, property.name()):
                    self.emit(SIGNAL('valueChanged()'))
            else:
                property.setHighlighted(True)
    
    def itemDoubleClickedSlot(self, item):
        """ Slot for itemClicked() signal.
        
        Calls items's property's doubleClicked().
        """
        logging.debug(self.__class__.__name__ + ": itemDoubleClickedSlot()")
        if item.property():
            item.property().labelDoubleClicked()


class LabelItem(QTableWidgetItem):
    """ A QTableWidgetItem with a convenient constructor. 
    """
    def __init__(self, argument, color=Qt.white):
        """ Constructor.
        
        Argument may be either a string or a Property object.
        If argument is the latter the property's user info will be used for the label's tooltip.
        """
        if isinstance(argument, Property):
            tooltip = argument.name() + " (" + argument.userInfo() + ")"
            name = argument.name()
            self._property = argument
        else:
            tooltip = argument
            name = argument
            self._property = None
            
        QTableWidgetItem.__init__(self, name)
        self.setToolTip(tooltip)
        self.setFlags(Qt.ItemIsEnabled)
        self.setBackgroundColor(color)
    
    def property(self):
        return self._property

class Property(object):
    """ Mother of all properties which can be added to the PropertyView using its append() function.
    """
    
    USER_INFO = "General property"
    DEFAULT_HEIGHT = 20
    
    def __init__(self, name):
        self.setName(name)
        self._propertyView = None
        
    def setName(self, name):
        """ Sets the name of this property.
        """
        self._name = name
    
    def name(self):
        """ Return the name of this property.
        """
        return self._name
    
    def setPropertyView(self, propertyView):
        """ Sets PropertyView object.
        """
        self._propertyView = propertyView
        
    def propertyView(self):
        """ Returns property view.
        """
        return self._propertyView
    
    def userInfo(self):
        """ Returns user info string containing information on type of property and what data may be insert.
        """
        return self.USER_INFO
    
    def setReadOnly(self, readOnly):
        """ Disables editing functionality.
        """
        pass
    
    def properyHeight(self):
        """ Return the height of the property widget.
        """
        return self.DEFAULT_HEIGHT
    
    def setValue(self, value):
        """ Abstract function returning current value of this property.
        
        Has to be implemented by properties which allow the user to change their value.
        """
        raise NotImplementedError
    
    def value(self):
        """ Abstract function returning current value of this property.
        
        Has to be implemented by properties which allow the user to change their value.
        """
        raise NotImplementedError
    
    def valueChanged(self):
        """ Slot for change events. 
        
        The actual object which have changed should connect their value changed signal 
        (or similar) to this function to forward change to data accessor of PropertyView.
        """
        logging.debug('Property: valueChanged() ' + str(self.name()))
        if self.propertyView():
            self.propertyView().valueChanged(self)

    def labelDoubleClicked(self):
        """ Called by PropertyView itemDoubleClicked().
        """
        pass
    
    def setHighlighted(self,highlight):
        """ Highlight the property, e.g. change color.
        """
        pass

class BooleanProperty(Property, QCheckBox):
    """ Property holding a check box for boolean values.
    """
    
    USER_INFO = "Enable / Disable"
    
    def __init__(self, name, value):
        """ Constructor.
        """
        Property.__init__(self, name)
        QCheckBox.__init__(self)
        self.connect(self, SIGNAL('stateChanged(int)'), self.valueChanged)
    
    def setReadOnly(self, readOnly):
        """ Disables editing functionality.
        """
        if readOnly and self.isCheckable():
            self.setCheckable(False)
            self.disconnect(self, SIGNAL('stateChanged(int)'), self.valueChanged)
        elif not readOnly and not self.isCheckable():
            self.setCheckable(True)
            self.connect(self, SIGNAL('stateChanged(int)'), self.valueChanged)
        
    def value(self):
        """ Returns True if check box is checked.
        """
        return self.isChecked()
        
class TextEditWithButtonProperty(Property, QWidget):
    """
    This class provides a PropertyView property holding an editable text and a button.
    It is possible to hide the button unless the mouse cursor is over the property. This feature is turned on by default. See setAutohideButton().
    If the button is pressed nothing happens. This functionality should be implemented in sub-classes. See buttonClicked().
    """
    
    BUTTON_LABEL = '...'
    AUTOHIDE_BUTTON = True
    
    def __init__(self, name, value, multiline=False):
        """ The constructor creates a QHBoxLayout and calls createTextEdit() and createButton(). 
        """
        Property.__init__(self, name)
        QWidget.__init__(self)
        self._textEdit = None
        self._button = None
        self.setAutohideButton(self.AUTOHIDE_BUTTON)
        
        self.setContentsMargins(0, 0, 0, 0)
        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)
        
        self._readOnly = False
        self._multiline = multiline
        self._value = ""

        self.createTextEdit(value)
        self.createButton()
        
    def setValue(self, value):
        """ Sets value of text edit.
        """
        if self._textEdit:
            self._value=value
            self._textEdit.setText(self._value)
            # TODO: sometimes when changing value the text edit appears to be empty when new text is shorter than old text
            #if not self._multiline:
            #    self._textEdit.setCursorPosition(self._textEdit.displayText().length())
                
        
    def createTextEdit(self, value=None):
        """ This function creates the text field and adds it to the property's layout. 
        
        Single line (QLineEdit) as well as Multiline
        """
        if not self._multiline:
            self._textEdit = QLineEdit(self)
        else:
            self._textEdit = QTextEdit(self)
        if value != None:
            value = str(value)
        else:
            value = ""
        self.setValue(value)
        if not self._multiline:
            self._textEdit.setFrame(False)
            self.connect(self._textEdit, SIGNAL('editingFinished()'), self.valueChanged)
        else:
            self._textEdit.setWordWrapMode(QTextOption.NoWrap)
            self._textEdit.setFrameStyle(QFrame.NoFrame)
            self.connect(self._textEdit, SIGNAL('textChanged()'), self.valueChanged)
            
        self._textEdit.setToolTip(value)
        self._textEdit.setContentsMargins(0, 0, 0, 0)
        self._textEdit.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding))
        self.layout().addWidget(self._textEdit)
        
    def properyHeight(self):
        if self._multiline:
            self._textEdit.document().adjustSize()
            height=self._textEdit.document().size().height()+3
            if self._textEdit.horizontalScrollBar().isVisible():
                height+=self._textEdit.horizontalScrollBar().height()+3
            return height
        else:
            return self.DEFAULT_HEIGHT
            
    def textEdit(self):
        """ Returns text edit.
        """
        return self._textEdit
        
    def hasTextEdit(self):
        """ Returns True if the text field has been created, otherwise False is returned. 
        """
        return self._textEdit != None
            
    def createButton(self):
        """ Creates a button and adds it to the property's layout.
        """
        self._button = QToolButton(self)
        self._button.setText(self.BUTTON_LABEL)
        self._button.setContentsMargins(0, 0, 0, 0)
        self.connect(self._button, SIGNAL('clicked(bool)'), self.buttonClicked)
        self.layout().addWidget(self._button)
        
        if self.autohideButtonFlag:
            self._button.hide()
    
    def button(self):
        """ Return button.
        """
        return self._button
    
    def hasButton(self):
        """ Returns True if the button has been created, otherwise False is returned. 
        """
        return self._button != None
    
    def setReadOnly(self, readOnly):
        self._readOnly = readOnly
        if not self.textEdit().isReadOnly() and readOnly:
            self.textEdit().setReadOnly(readOnly)
            if not self._multiline:
                self.disconnect(self._textEdit, SIGNAL('editingFinished()'), self.valueChanged)
            else:
                self.disconnect(self._textEdit, SIGNAL('textChanged()'), self.valueChanged)
            if self.hasButton():
                del self._button
        if self.textEdit().isReadOnly() and not readOnly:
            if not self._multiline:
                self.connect(self._textEdit, SIGNAL('editingFinished()'), self.valueChanged)
            else:
                self.connect(self._textEdit, SIGNAL('textChanged()'), self.valueChanged)
            self.textEdit().setReadOnly(readOnly)
            self.createButton()
    
    def readOnly(self):
        return self._readOnly
    
    def value(self):
        """ Returns value of text edit.
        """
        if self.hasTextEdit():
            if not self._multiline:
                return str(self._textEdit.text())
            else:
                return str(self._textEdit.toPlainText())
        return None
    
    def setAutohideButton(self, hide):
        """ If hide is True the button will only be visible while the cursor is over the property. 
        """
        self.autohideButtonFlag = hide
        
    def buttonClicked(self, checked=False):
        """
        This function is called if the button was clicked. For information on the checked argument see documentation of QPushButton::clicked().
        This function should be overwritten by sub-classes.
        """ 
        True
        
    def enterEvent(self, event):
        """ If autohideButtonFlag is set this function makes the button visible. See setAutohideButton(). 
        """
        if self.autohideButtonFlag and self.hasButton():
            self._button.show()
            
    def leaveEvent(self, event):
        """ If autohideButtonFlag is set this function makes the button invisible. See setAutohideButton(). 
        """
        if self.autohideButtonFlag and self.hasButton():
            self._button.hide()

    def valueChanged(self):
        Property.valueChanged(self)
        self.emit(SIGNAL('updatePropertyHeight'),self)
        self._textEdit.setToolTip(self.value())
        
    def setHighlighted(self,highlight):
        """ Highlight the property by changing the background color of the textfield.
        """
        p=QPalette()
        if highlight:
            p.setColor(QPalette.Active, QPalette.ColorRole(9),Qt.red)
        else:
            p.setColor(QPalette.Active, QPalette.ColorRole(9),Qt.white)
        if not self._multiline:
            self._textEdit.setPalette(p)
        else:
            self._textEdit.viewport().setPalette(p)
    
    def keyPressEvent(self,event):
        QWidget.keyPressEvent(self,event)
        if event.key()==Qt.Key_Escape:
            self.setValue(self._value)

class StringProperty(TextEditWithButtonProperty):
    """ This property only holds an editable text line. 
    """
    
    USER_INFO = "Text field"
    
    AUTHIDE_BUTTON = False
    
    def __init__(self, name, value):
        """ Constructor """
        TextEditWithButtonProperty.__init__(self, name, value)
        
    def createButton(self):
        """ Do not create a button."""
        pass
    
class TextProperty(TextEditWithButtonProperty):
    """ This property only holds an editable text field. 
    """
    
    USER_INFO = "Text field"
    
    AUTHIDE_BUTTON = False
    
    def __init__(self, name, value):
        """ Constructor """
        TextEditWithButtonProperty.__init__(self, name, value, True)
        
    def createButton(self):
        """ Do not create a button."""
        pass
    
class IntegerProperty(StringProperty):
    """ StringProperty which only accepts integer numbers.
    """
    
    USER_INFO = "Integer field"
    
    def __init__(self, name, value):
        """ Constructor
        """
        StringProperty.__init__(self, name, value)
        
    def createTextEdit(self, value=None):
        StringProperty.createTextEdit(self, value)
        self.textEdit().setInputMask("0000000000000000000000000000")
            
class FileProperty(TextEditWithButtonProperty):
    """ This property has an editable text line and a button. If the button is clicked a dialog allowing to chose a file appears. """
    
    USER_INFO = "Select a file. Double click on label to open file."
    
    _recentFile = ''
    
    def __init__(self, name, value):
        TextEditWithButtonProperty.__init__(self, name, value)
        self.button().setToolTip(self.USER_INFO)
        
    def buttonClicked(self, checked=False):
        """ Shows the file selection dialog. """ 
#        filename = QFileDialog.getOpenFileName(
#                                               self,
#                                               'Select a file',
#                                               self._recentFile
#                                               )
        filename = QFileDialog.getSaveFileName(
                                               self,
                                               'Select a file',
                                               self.value(),
                                               '',
                                               None,
                                               QFileDialog.DontConfirmOverwrite)
        if not filename.isEmpty():
            self._recentFile = filename
            self.setValue(filename)
            self.textEdit().emit(SIGNAL('editingFinished()'))
            
    def labelDoubleClicked(self):
        """ Open selected file in default application.
        """
        if isinstance(self.propertyView().parent(), AbstractTab):
            self.propertyView().parent().mainWindow().application().doubleClickOnFile(self.value())
