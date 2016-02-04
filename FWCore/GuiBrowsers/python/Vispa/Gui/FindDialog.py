import logging

from PyQt4.QtCore import QCoreApplication,Qt,SIGNAL
from PyQt4.QtGui import QDialog,QLabel,QLineEdit,QCheckBox,QPushButton,QVBoxLayout,QHBoxLayout,QMessageBox,QToolButton,QWidget,QLayout

from Vispa.Main.Application import Application
from Vispa.Share.ThreadChain import ThreadChain

class FindDialog(QDialog):
    def __init__(self,parent=None):
        logging.debug(__name__ +': __init__')
        QDialog.__init__(self,parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Find...")
        
        self._findAlgorithm=None
        self._properties=[]
        self._scripts=[]
        self._find=True
        self._filter=False

        self.fill()
        
    def fill(self):
        logging.debug(__name__ +': fill')
        self._findLabelLabel = QLabel("Label: ")
        self._findLabelLineEdit = QLineEdit()
        self._findLabelLineEdit.setToolTip("Example: Particle1")

        self._caseSensitiveCheckBox=QCheckBox("Case sensitive")
        self._exactMatchCheckBox=QCheckBox("Exact match")
        self._helpButton = QPushButton("&Help")
        
        self._findPreviousButton = QPushButton("&Previous")
        self._findPreviousButton.hide()
        self._findNumberLabel = QLabel("?/?")
        self._findNumberLabel.hide()
        self._findNextButton = QPushButton("&Find")
        self._filterButton = QPushButton("&Filter")
        self._resetButton = QPushButton("&Reset")
        self._closeButton = QPushButton("&Close")

        self.setLayout(QVBoxLayout())
        self.layout().setSizeConstraint(QLayout.SetFixedSize)
        self._layout1=QHBoxLayout()
        self._layout3=QHBoxLayout()
        self._layout4=QHBoxLayout()
    
        self._layout1.setSizeConstraint(QLayout.SetDefaultConstraint)
        self._layout3.setSizeConstraint(QLayout.SetDefaultConstraint)
        self._layout4.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.layout().addLayout(self._layout1)
        self.layout().addLayout(self._layout3)
        self.layout().addStretch()
        self.layout().addLayout(self._layout4)
        
        self._layout1.addWidget(self._findLabelLabel)
        self._layout1.addWidget(self._findLabelLineEdit)

        self._layout3.addWidget(self._helpButton)
        self._layout3.addStretch()
        self._layout3.addWidget(self._caseSensitiveCheckBox)
        self._layout3.addWidget(self._exactMatchCheckBox)
        
        self._layout4.addWidget(self._findPreviousButton)
        self._layout4.addWidget(self._findNumberLabel)
        self._layout4.addWidget(self._findNextButton)
        self._layout4.addWidget(self._filterButton)
        self._layout4.addWidget(self._resetButton)
        self._layout4.addStretch()
        self._layout4.addWidget(self._closeButton)

        self.connect(self._findLabelLineEdit, SIGNAL('textChanged(QString)'), self.edited)
        self.connect(self._caseSensitiveCheckBox, SIGNAL('stateChanged(int)'), self.edited)
        self.connect(self._exactMatchCheckBox, SIGNAL('stateChanged(int)'), self.edited)

        self.connect(self._findPreviousButton, SIGNAL('clicked(bool)'), self.findPrevious)
        self.connect(self._findNextButton, SIGNAL('clicked(bool)'), self.findNext)
        self.connect(self._filterButton, SIGNAL('clicked(bool)'), self.filter)
        self.connect(self._resetButton, SIGNAL('clicked(bool)'), self.reset)
        self.connect(self._helpButton, SIGNAL('clicked(bool)'), self.help)
        self.connect(self._closeButton, SIGNAL('clicked(bool)'), self.reject)

        self._addStringProperty(False,False)
        self._addScript(False,False)

    def _removeProperty(self):
        for property in self._properties:
            if self.sender() in property:
                self._remove(property)
                return

    def _remove(self,object):
        for o in object:
            if isinstance(o,QWidget):
                o.close()
        self.layout().removeItem(object[0])
        if object in self._properties:
            self._properties.remove(object)
        elif object in self._scripts:
            self._scripts.remove(object)

    def _addStringProperty(self,bool,deletable=True):

        layout2=QHBoxLayout()

        findPropertyNameLabel = QLabel("Property: ")
        findPropertyNameLineEdit = QLineEdit()
        findPropertyNameLineEdit.setToolTip("Example: Label = Particle1 ")
        findPropertyValueLabel = QLabel(" = ")
        findPropertyValueLineEdit = QLineEdit()
        findPropertyValueLineEdit.setToolTip("Example: Label = Particle1 ")
        propertyAdd = QToolButton()
        propertyAdd.setText("+")
        propertyDelete = QToolButton()
        propertyDelete.setText("-")
        
        if deletable:
            propertyAdd.hide()
        else:
            propertyDelete.hide()
        layout2.addWidget(propertyAdd)
        layout2.addWidget(propertyDelete)
        layout2.addWidget(findPropertyNameLabel)
        layout2.addWidget(findPropertyNameLineEdit)
        layout2.addWidget(findPropertyValueLabel)
        layout2.addWidget(findPropertyValueLineEdit)

        self.connect(findPropertyNameLineEdit, SIGNAL('textChanged(QString)'), self.edited)
        self.connect(findPropertyValueLineEdit, SIGNAL('textChanged(QString)'), self.edited)
        self.connect(propertyAdd, SIGNAL('clicked(bool)'), self._addStringProperty)
        self.connect(propertyDelete, SIGNAL('clicked(bool)'), self._removeProperty)

        self.layout().insertLayout(len(self._properties)+len(self._scripts)+1,layout2)
        
        self._properties+=[(layout2,findPropertyNameLineEdit,findPropertyValueLineEdit,findPropertyNameLabel,findPropertyValueLabel,propertyAdd,propertyDelete)]
        
    def _removeScript(self):
        for script in self._scripts:
            if self.sender() in script:
                self._remove(script)
                return
        
    def _addScript(self,bool,deletable=True):

        layout2=QHBoxLayout()

        findScriptLabel = QLabel("Filter = ")
        findScriptLineEdit = QLineEdit("")
        findScriptLineEdit.setToolTip("Example: object.Label == 'Particle1' ")
        scriptAdd = QToolButton()
        scriptAdd.setText("+")
        scriptDelete = QToolButton()
        scriptDelete.setText("-")
        
        if deletable:
            scriptAdd.hide()
        else:
            scriptDelete.hide()
        layout2.addWidget(scriptAdd)
        layout2.addWidget(scriptDelete)
        layout2.addWidget(findScriptLabel)
        layout2.addWidget(findScriptLineEdit)

        self.connect(findScriptLineEdit, SIGNAL('textChanged(QString)'), self.edited)
        self.connect(scriptAdd, SIGNAL('clicked(bool)'), self._addScript)
        self.connect(scriptDelete, SIGNAL('clicked(bool)'), self._removeScript)

        self.layout().insertLayout(len(self._properties)+len(self._scripts)+1,layout2)
        
        self._scripts+=[(layout2,findScriptLineEdit,findScriptLabel,scriptAdd,scriptDelete)]
        
    def onScreen(self, filter=False, find=True):
        logging.debug(__name__ +': onScreen')
        self._find=find
        self._filter=filter
        if self._find and self._filter:
            self._findNextButton.setDefault(True)
            self.setWindowTitle("Find/Filter...")
        elif self._find:
            self._findNextButton.setDefault(True)
            self.setWindowTitle("Find...")
        elif self._filter:
            self._filterButton.setDefault(True)
            self.setWindowTitle("Filter...")
            
        self._findNextButton.setVisible(find)
        if not find:
            self._findPreviousButton.setVisible(find)
        self._filterButton.setVisible(filter)
        self.show()
        self.raise_()
        self.activateWindow()
        self._findLabelLineEdit.setFocus()
        
    def keyPressEvent(self, event):
        """ 
        """
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_W:
            self.close()
        QDialog.keyPressEvent(self, event)

    def setFindAlgorithm(self,findAlgorithm):
        logging.debug(__name__ +': setFindAlgorithm')
        self._findAlgorithm=findAlgorithm
        
    def findAlgorithm(self):
        return self._findAlgorithm
        
    def label(self):
        return str(self._findLabelLineEdit.text().toAscii())
    
    def setLabel(self,label):
        logging.debug(__name__ +': setLabel '+label)
        self._findLabelLineEdit.setText(label)
    
    def properties(self):
        return [(str(property[1].text().toAscii()),str(property[2].text().toAscii())) for property in self._properties]
    
    def scripts(self):
        return [str(script[1].text().toAscii()) for script in self._scripts]
    
    def caseSensitive(self):
        return self._caseSensitiveCheckBox.checkState()==Qt.Checked

    def exactMatch(self):
        return self._exactMatchCheckBox.checkState()==Qt.Checked
    
    def edited(self):
        self._findPreviousButton.hide()
        if self._findNextButton.isVisible():
            self._findNumberLabel.hide()
            self._findNextButton.setText("&Find")
    
    def _updateNumberLabel(self):
        current=self._findAlgorithm.currentNumber()
        total=self._findAlgorithm.numberOfResults()
        message=self._findAlgorithm.message()
        text=""
        if self._filter:
            text=str(total)+" found"
        else:
            if total>0:
                text=str(current)+"/"+str(total)
            else:
                text="not found"
        if message:
            text+=" ("+message+")"
        self._findNumberLabel.setText(text)
    
    def findPrevious(self):
        logging.debug(__name__ +': findPrevious')
        object=self._findAlgorithm.previous()
        self._updateNumberLabel()
        self.emit(SIGNAL("found"),object)
    
    def findNext(self):
        logging.debug(__name__ +': findNext')
        if not self._findPreviousButton.isVisible():
            self._findNextButton.setVisible(False)
            self._filterButton.setVisible(False)
            self._resetButton.setVisible(False)
            self._findNumberLabel.setText("Searching...")
            self._findNumberLabel.show()
            thread = ThreadChain(self._findAlgorithm.findUsingFindDialog, self)
            while thread.isRunning():
                if not Application.NO_PROCESS_EVENTS:
                    QCoreApplication.instance().processEvents()
            object=thread.returnValue()
            self._findNextButton.setVisible(True)
            if self._filter:
                self._filterButton.setVisible(True)
            self._resetButton.setVisible(True)
            self._findPreviousButton.show()
            self._findNextButton.setText("&Next")
        else:
            object=self._findAlgorithm.next()
        self._updateNumberLabel()
        self.emit(SIGNAL("found"),object)

    def filter(self):
        logging.debug(__name__ +': filter')
        self._findNextButton.setVisible(False)
        self._filterButton.setVisible(False)
        self._resetButton.setVisible(False)
        self._findNumberLabel.setText("Searching...")
        self._findNumberLabel.show()
        thread = ThreadChain(self._findAlgorithm.findUsingFindDialog, self)
        while thread.isRunning():
            if not Application.NO_PROCESS_EVENTS:
                QCoreApplication.instance().processEvents()
        if self._find:
            self._findNextButton.setVisible(True)
        self._filterButton.setVisible(True)
        self._resetButton.setVisible(True)
        self._updateNumberLabel()
        self.emit(SIGNAL("filtered"),self._findAlgorithm.results())

    def reset(self):
        self.setLabel("")
        for o in self._scripts+self._properties:
            self._remove(o)
        self._addStringProperty(False,False)
        self._addScript(False,False)
        self._findAlgorithm.clear()
        self._updateNumberLabel()
        if self._filter:
            self.emit(SIGNAL("filtered"),None)
        self.update()

    def help(self):
        QMessageBox.about(self, 'Info', "You can find objects \n1. using their label shown in the center view, \n2. their properties shown in the property view, or \n3. using a Python script returning a boolean. Empty fields are ignored. Examples are shown as tool tips.")
