import logging

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Vispa.Main.Thread import RunThread

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

        self._caseSensitiveCheckBox=QCheckBox("Case sensitive")
        self._exactMatchCheckBox=QCheckBox("Exact match")
        self._addStringPropertyButton=QPushButton("&Add Property")
        self._addScriptButton=QPushButton("&Add Script")
        
        self._findPreviousButton = QPushButton("&Previous")
        self._findPreviousButton.hide()
        self._findNumberLabel = QLabel("?/?")
        self._findNumberLabel.hide()
        self._findNextButton = QPushButton("&Find")
        self._filterButton = QPushButton("&Filter")
        self._resetButton = QPushButton("&Reset")
        self._closeButton = QPushButton("&Close")

        self.setLayout(QVBoxLayout())
        self._layout1=QHBoxLayout()
        self._layout3=QHBoxLayout()
        self._layout4=QHBoxLayout()
    
        self.layout().addLayout(self._layout1)
        self.layout().addLayout(self._layout3)
        self.layout().addStretch()
        self.layout().addLayout(self._layout4)
        
        self._layout1.addWidget(self._findLabelLabel)
        self._layout1.addWidget(self._findLabelLineEdit)

        self._layout3.addWidget(self._addStringPropertyButton)
        self._layout3.addWidget(self._addScriptButton)
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

        self.connect(self._addStringPropertyButton, SIGNAL('clicked(bool)'), self._addStringProperty)
        self.connect(self._addScriptButton, SIGNAL('clicked(bool)'), self._addScript)
        self.connect(self._findPreviousButton, SIGNAL('clicked(bool)'), self.findPrevious)
        self.connect(self._findNextButton, SIGNAL('clicked(bool)'), self.findNext)
        self.connect(self._filterButton, SIGNAL('clicked(bool)'), self.filter)
        self.connect(self._resetButton, SIGNAL('clicked(bool)'), self.reset)
        self.connect(self._closeButton, SIGNAL('clicked(bool)'), self.reject)

        self._addStringProperty()
        self._addScript()

    def _removeProperty(self):
        if len(self._properties)==0:
            return False
        property=self._properties[len(self._properties)-1]
        property[1].close()
        property[2].close()
        property[3].close()
        property[4].close()
        self.layout().removeItem(property[0])
        self._properties.remove(property)
        return True

    def _addStringProperty(self):

        layout2=QHBoxLayout()

        findPropertyNameLabel = QLabel("Property: ")
        findPropertyNameLineEdit = QLineEdit()
        findPropertyValueLabel = QLabel(" = ")
        findPropertyValueLineEdit = QLineEdit()
        
        layout2.addWidget(findPropertyNameLabel)
        layout2.addWidget(findPropertyNameLineEdit)
        layout2.addWidget(findPropertyValueLabel)
        layout2.addWidget(findPropertyValueLineEdit)

        self.connect(findPropertyNameLineEdit, SIGNAL('textChanged(QString)'), self.edited)
        self.connect(findPropertyValueLineEdit, SIGNAL('textChanged(QString)'), self.edited)

        self.layout().insertLayout(len(self._properties)+len(self._scripts)+1,layout2)
        
        self._properties+=[(layout2,findPropertyNameLineEdit,findPropertyValueLineEdit,findPropertyNameLabel,findPropertyValueLabel)]
        
    def _removeScript(self):
        if len(self._scripts)==0:
            return False
        script=self._scripts[len(self._scripts)-1]
        script[1].close()
        script[2].close()
        self.layout().removeItem(script[0])
        self._scripts.remove(script)
        return True

    def _addScript(self):

        layout2=QHBoxLayout()

        findScriptLabel = QLabel("Filter = ")
        findScriptLineEdit = QLineEdit("")
        findScriptLineEdit.setToolTip("Example filter: object.Label == 'example' ")
        
        layout2.addWidget(findScriptLabel)
        layout2.addWidget(findScriptLineEdit)

        self.connect(findScriptLineEdit, SIGNAL('textChanged(QString)'), self.edited)

        self.layout().insertLayout(len(self._properties)+len(self._scripts)+1,layout2)
        
        self._scripts+=[(layout2,findScriptLineEdit,findScriptLabel)]
        
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
        return str(self._findLabelLineEdit.text())
    
    def setLabel(self,label):
        logging.debug(__name__ +': setLabel '+label)
        self._findLabelLineEdit.setText(label)
    
    def properties(self):
        return [(str(property[1].text()),str(property[2].text())) for property in self._properties]
    
    def scripts(self):
        return [str(script[1].text()) for script in self._scripts]
    
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
            thread = RunThread(self._findAlgorithm.findUsingFindDialog, self)
            while thread.isRunning():
                QCoreApplication.instance().processEvents()
            object=thread.returnValue
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
        logging.debug(__name__ +': findNext')
        self._findNextButton.setVisible(False)
        self._filterButton.setVisible(False)
        self._resetButton.setVisible(False)
        self._findNumberLabel.setText("Searching...")
        self._findNumberLabel.show()
        thread = RunThread(self._findAlgorithm.findUsingFindDialog, self)
        while thread.isRunning():
            QCoreApplication.instance().processEvents()
        if self._find:
            self._findNextButton.setVisible(True)
        self._filterButton.setVisible(True)
        self._resetButton.setVisible(True)
        self._updateNumberLabel()
        self.emit(SIGNAL("filtered"),self._findAlgorithm.results())

    def reset(self):
        self.setLabel("")
        for p in self._properties:
            self._removeProperty()
        for p in self._scripts:
            self._removeScript()
        self._addStringProperty()
        self._addScript()
        self._findAlgorithm.clear()
        self._updateNumberLabel()
        if self._filter:
            self.emit(SIGNAL("filtered"),None)
