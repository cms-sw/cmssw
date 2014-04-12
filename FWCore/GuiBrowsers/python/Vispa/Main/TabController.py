import os.path
import logging
import math

from PyQt4.QtCore import QObject
from PyQt4.QtGui import QMessageBox,QInputDialog

from Vispa.Main.Filetype import Filetype
from Vispa.Share.UndoEvent import UndoEvent

class TabController(QObject):
    """ Base class for all tab controllers.
    
    Tab controllers control the functionality of plugin tabs.
    """ 
    
    TAB_LABEL_MAX_LENGTH = 20

    def __init__(self, plugin):
        QObject.__init__(self)
        logging.debug(__name__ + ": __init__")
        self._plugin = plugin
        self._fileModifiedFlag = False
        self._isEditableFlag = True
        self._tab = None
        self._filename = None
        self._copyPasteEnabledFlag = False
        self._allowSelectAllFlag = False
        self._findEnabledFlag = False
        self._userZoomLevel = 100
        self._zoomButtonPressedBeforeFlag = False
        self._fileModifcationTimestamp = None
        self._showingModifiedMessageFlag=False
        self._supportsUndo = False
        self._undoEvents = []
        self._redoEvents = []
           
    #@staticmethod
    def staticSupportedFileTypes():
        """ Static function returning all filetypes the tab controller can handle.
        
        Sub classes should reimplement this function. It returns a list with 2-tuples of the following form:
        ('extension', 'description of file type').
        """
        return []
    staticSupportedFileTypes = staticmethod(staticSupportedFileTypes)

    def supportedFileTypes(self):
        """ Returns staticSupportedFileTypes() of the class to which this object belongs.
        """
        return self.__class__.staticSupportedFileTypes()
    
    def supportedFileFilters(self):
        supportedFileTypes = self.__class__.staticSupportedFileTypes()
        return ";;".join([Filetype(t[0], t[1]).fileDialogFilter() for t in supportedFileTypes])
            
    def plugin(self):
        """ Returns the plugin reference, set by setPlugin().
        """
        return self._plugin
       
    def setTab(self, tab):
        """ Sets tab.
        """
        self._tab = tab
        
    def tab(self):
        """ Returns tab.
        """
        return self._tab
    
    def setFilename(self, filename):
        """ Sets a filename.
        """
        self._filename = filename
        if self._filename and os.path.exists(self._filename):
            self._fileModifcationTimestamp = os.path.getmtime(self._filename)
    
    def filename(self):
        """ Returns filename of this tab.
        """
        return self._filename
    
    def getFileBasename(self):
        """ Returns the basename of this tab's filename.
        
        Part of filename after last /.
        """
        return os.path.basename(self._filename) 
    
    def setCopyPasteEnabled(self, enable=True):
        """ Sets a flag indicating whether this tab can handle copy and paste events.
        
        See also isCopyPasteEnabled(), cut(), copy(), paste().
        """
        self._copyPasteEnabledFlag = enable
        
    def isCopyPasteEnabled(self):
        """ Return True if the copyPasteFlag is set.
        
        See setCopyPasteEnabled(), cut(), copy(), paste().
        """
        return self._copyPasteEnabledFlag
    
    def setFindEnabled(self, enable=True):
        """Sets a flag indicating whether this tab can handle find requests.
        
        See isFindEnabled(), find().
        """
        self._findEnabledFlag = enable
        
    def isFindEnabled(self):
        """Returns True if findEnabledFlag is set.
        
        See setFindEnabled(), find().
        """
        return self._findEnabledFlag
    
    def updateLabel(self,prefix="",titletext = ""):
        """ Sets the text of the tab to filename if it is set. If
        titletext
        is not an emty string, it is used instead of the filename.
        
        Otherwise it is set to 'UNTITLED'. It also evaluates the fileModifiedFlag and indicates changes with an *.
        """
        if self._filename:
            title = os.path.basename(self._filename)
            if len(os.path.splitext(title)[0]) > self.TAB_LABEL_MAX_LENGTH:
                ext = os.path.splitext(title)[1].lower().strip(".")
                title = os.path.splitext(title)[0][0:self.TAB_LABEL_MAX_LENGTH] + "...." + ext
        elif titletext == "":
            title = 'UNTITLED'
        else:
            title = titletext
        
        if self.isModified():
            title = '*' + title
        
        title = prefix+title
        
        if self.tab().tabWidget():
            self.tab().tabWidget().setTabText(self.tab().tabWidget().indexOf(self.tab()), title)
        else:
            self.tab().setWindowTitle(title)
        
    def setModified(self, modified=True):
        """ Sets the file Modified flag to True or False.
        
        This affects the closing of this tab.
        It is only possible to set the modification flag to True if the controller is editable (see isEditable()).
        """
        if modified and not self.isEditable():
            return

        previous = self._fileModifiedFlag
        self._fileModifiedFlag = modified
        
        if previous != self._fileModifiedFlag:
            self.updateLabel()
            if self.tab().mainWindow():
                self.plugin().application().updateMenuAndWindowTitle()
            else:
                logging.info(self.__class__.__name__ +": setModified() - Cannot tell application the modification state: There is no application associated with the tab.")
        
    def isModified(self):
        """ Evaluates the file Modified flag. Always returns True if no filename is set.
        """
        return self._fileModifiedFlag
        
    def setEditable(self, editable):
        """ Sets the file Editable flag.
        """
        #logging.debug(self.__class__.__name__ +": setEditable(" + str(editable) +")")
        self._isEditableFlag = editable
        self.plugin().application().updateMenu()
        
    def isEditable(self):
        """ Evaluates the file Editable flag.
        """
        return self._isEditableFlag
    
    def setAllowSelectAll(self, allowSelectAll):
        """ Sets the allowSelectAll flag.
        """
        self._allowSelectAllFlag = allowSelectAll
        self.plugin().application().updateMenu()
        
    def allowSelectAll(self):
        """ Evaluates the sllowSelectAll flag.
        """
        return self._allowSelectAllFlag
        
    def open(self, filename=None, update=True):
        """ Open given file.
        """
        logging.debug(self.__class__.__name__ + ": open()")
        
        statusMessage = self.plugin().application().startWorking("Opening file " + filename)
            
        if filename == None:
            if self._filename:
                filename = self._filename
            else:
                self.plugin().application().stopWorking(statusMessage, "failed")
                return False
        
        if self.readFile(filename):
            self.setFilename(filename)
            self.updateLabel()
            if update:
	            self.updateContent()
            self.plugin().application().stopWorking(statusMessage)
            return True
        
        self.plugin().application().stopWorking(statusMessage, "failed")
        return False
    
    def readFile(self, filename):
        """
        This function performs the actual reading of a file. It should be overwritten by any PluginTab which inherits Tab.
        If the reading was successful True should be returned.
        The file should be read from the file given in the argument filename not to the one in self._filename.
        """
        raise NotImplementedError
    
    def save(self, filename=""):
        """ Takes care the tab's data will be written to the file given as argument or if this is an empty string to self._filename.
        
        Whenever the content of the tab should be saved, this method should be called. If no filename is specified nor already set set it asks the user to set one. 
        Afterwards the writing is initiated by calling writeFile(). 
        """
        #logging.debug('Tab: save()')
        
        if filename == '':
            if self._filename:
                filename = self._filename
            else:
                return self.plugin().application().saveFileAsDialog()
        
        statusMessage = self.plugin().application().startWorking("Saving file " + filename)

        good=True
        message=""
        try:  
            good=self.writeFile(filename)
        except Exception,e:
            good=False
            message="\n"+str(e)
        if good:
            self.setFilename(filename)
            self.setModified(False)
            self.updateLabel()
            self.plugin().application().addRecentFile(filename)
            self.plugin().application().updateMenuAndWindowTitle()
            
            # set last saved state for undo events
            if len(self._redoEvents) > 0:
                lastSavedStateEvent = self._redoEvents[len(self._redoEvents) -1]
            else:
                lastSavedStateEvent = None
            self.setLastSavedStateEvent(lastSavedStateEvent)
            
            self.plugin().application().stopWorking(statusMessage)
            return True
        else:
            QMessageBox.critical(self.tab().mainWindow(), 'Error while saving data', 'Could not write to file ' + filename +'.'+message)
            logging.error(self.__class__.__name__ + ": save() : Could not write to file " + filename +'.'+message)
            self.plugin().application().stopWorking(statusMessage, "failed")
            return False

    def allowClose(self):
        if self.isEditable() and self.isModified():
            messageResult = self.plugin().application().showMessageBox("The document has been modified.",
                                                                       "Do you want to save your changes?",
                                                                       QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                                                       QMessageBox.Save)
            
            if messageResult == QMessageBox.Save:
                if not self.save():
                    return False
            elif messageResult == QMessageBox.Cancel:
                return False
        return True
    
    def close(self):
        """ Asks user if he wants to save potentially unsaved data and closes the tab. 
        
        This function usually does not need to be overwritten by a PluginTab.
        """ 
        allowClose = self.allowClose()
        if allowClose:
            if self.tab().tabWidget():
                self.tab().tabWidget().removeTab(self.tab().tabWidget().indexOf(self.tab()))
            #if self.tab() in self.tab().mainWindow()._tabWindows:
            #    self.tab().mainWindow()._tabWindows.remove(self.tab())
            self.tab().close()
            # no effect?
            #self._tab.deleteLater()
            #self._tab = None
        return allowClose
        
    def writeFile(self, filename):
        """
        This function performs the actual writing / saving of a file. It should be overwritten by any PluginTab which inherits Tab.
        If the writing was successful True should be returned.
        The file should be written to the file given in the argument filename not to the one in self._filename.
        These variables may differ in case the user selects "save as..." and picks a new filename on a file which already has a name set.
        If writing was successful the self._filename variable will then be set to the value of filename.
        """
        raise NotImplementedError
    
    def checkModificationTimestamp(self):
        """ Compares the actual modification timestamp of self.filename() to the modification at opening or last save operation.
        
        This function is called by Application when the tab associated with this controller was activated.
        If modification timestamps differ the refresh() method is called.
        """
        if not self._filename or self._fileModifcationTimestamp == 0:
            return
        
        msgBox = None
        if not os.path.exists(self._filename):
            logging.debug(self.__class__.__name__ + ": checkModificationTimestamp() - File was removed.")
            self._fileModifcationTimestamp = 0
            msgBox = QMessageBox()
            msgBox.setText("The file was removed.")
            if self.isEditable():
                msgBox.setInformativeText("Do you want to save the file with your version?")
                saveButton = msgBox.addButton("Save", QMessageBox.ActionRole)
                ignoreButton = msgBox.addButton("Ignore", QMessageBox.RejectRole)
            else:
                ignoreButton = msgBox.addButton("OK", QMessageBox.RejectRole)
            reloadButton = None
            
        elif self._fileModifcationTimestamp != os.path.getmtime(self._filename):
            logging.debug(self.__class__.__name__ + ": checkModificationTimestamp() - File was modified.")
            msgBox = QMessageBox()
            msgBox.setText("The file has been modified.")
            if self.isEditable():
                msgBox.setInformativeText("Do you want to overwrite the file with your version or reload the file?")
                saveButton = msgBox.addButton("Overwrite", QMessageBox.ActionRole)
            else:
                msgBox.setInformativeText("Do you want to reload the file?")
            reloadButton = msgBox.addButton("Reload", QMessageBox.DestructiveRole)
            ignoreButton = msgBox.addButton("Ignore", QMessageBox.RejectRole)
        
        if msgBox and not self._showingModifiedMessageFlag:
            self.setModified()
            self._showingModifiedMessageFlag=True
            msgBox.exec_()
            self._showingModifiedMessageFlag=False
            
            if self.isEditable() and msgBox.clickedButton() == saveButton:
                self.save()
            elif msgBox.clickedButton() == reloadButton:            
                self.refresh()
                self.setModified(False)
            elif msgBox.clickedButton() == ignoreButton and os.path.exists(self._filename):
                self._fileModifcationTimestamp = os.path.getmtime(self._filename)
                
        else:
            #logging.debug(self.__class__.__name__ + ": checkModificationTimestamp() - File was not modified.")
            pass
    
    def activated(self):
        """ Called by application when tab is activated in tabWidget.
        
        This function should be overwritten if special treatment on tab selection is required.
        """
        pass
        
    def cut(self):
        """ Handle cut event.
        
        This function is called if the user selects 'Cut' from menu. PluginTabs should override it if needed.
        See also setCopyPasteEnabled(), isCopyPasteEnabled().
        """
        raise NotImplementedError
        
    def copy(self):
        """ Handle copy event.
        
        This function is called if the user selects 'Copy' from menu. PluginTabs should override it if needed.
        See also setCopyPasteEnabled(), isCopyPasteEnabled().
        """
        raise NotImplementedError
        
    def paste(self):
        """ Handle paste event.
        
        This function is called if the user selects 'Paste' from menu. PluginTabs should override it if needed."
        See also setCopyPasteEnabled(), isCopyPasteEnabled().
        """
        raise NotImplementedError
        
    def find(self):
        """ Handle find event.
        
        This function is called if the user selects 'Find' from menu. PluginTabs should override it if needed."
        See also setFindEnabled(), isFindEnabled().
        """
        raise NotImplementedError
    
    def selectAll(self):
        """ Handle to perform select all action.
        
        This function is called if the user selects 'Select all' from menu. PluginTabs should override it if needed."
        See also setAllowSelectAll(), allowSelectAll().
        """
        raise NotImplementedError
    
    def setZoom(self, zoom):
        """ This function has to be implemented by tab controllers who want to use the zoom toolbar.
        
        The implementation has to forward the zoom value to the Zoomable object for which the toolbar is set up.
        See also zoom()
        """
        raise NotImplementedError
    
    def zoom(self):
        """ This function has to be implemented by tab controllers who want to use the zoom toolbar.
        
        The implementation should return the zoom value of the Zoomable object for which the toolbar is set up.
        See also setZoom()
        """
        raise NotImplementedError
          
    def zoomChanged(self, zoom):
        """ Shows zoom value on main window's status bar.
        """
        self.tab().mainWindow().statusBar().showMessage("Zoom " + str(round(zoom)) + " %")
    
    def resetZoomButtonPressedBefore(self):
        """ Sets the zoom button pressed before flag to False.
        
        If the flag is set functions handling the zoom toolbar buttons (zoomHundred(), zoomAll()) wont store the last zoom factor. The flag is set to true by these functions.
        By this mechanism the user can click the zoom buttons several times and will still be able to return to his orignal zoom level by zoomUser().
        The reset function needs to be called if the user manually sets the zoom level. For instance by connecting this function to the wheelEvent of the workspace scroll area.
        """
        self._zoomButtonPressedBeforeFlag = False
        
    def zoomUser(self):
        """ Returns to the manually set zoom factor before zoomHundred() or zoomAll() were called.
        """
        logging.debug(__name__ + ": zoomUser()")
        self.setZoom(self._userZoomLevel)
    
    def zoomHundred(self):
        """ Sets zoom factor to 100 %.
        """
        logging.debug(__name__ + ": zoomHundred()")
        if not self._zoomButtonPressedBeforeFlag:
            self._userZoomLevel = self.zoom()
            self._zoomButtonPressedBeforeFlag = True
        self.setZoom(100)
    
    def zoomAll(self):
        """ Zooms workspace content to fit optimal.
        
        Currently only works if scroll area is used and accessible through self.tab().scrollArea().
        """
        logging.debug(__name__ + ": zoomAll()")
        if not self._zoomButtonPressedBeforeFlag:
            self._userZoomLevel = self.zoom()
            self._zoomButtonPressedBeforeFlag = True
            
        viewportWidth = self.tab().scrollArea().viewport().width() 
        viewportHeight = self.tab().scrollArea().viewport().height()
        
        for i in range(0, 2):
            # do 2 iterations to prevent rounding error --> better fit
            workspaceChildrenRect = self.tab().scrollArea().widget().childrenRect()
            widthRatio = self.zoom() * viewportWidth / (workspaceChildrenRect.right())
            heightRatio = self.zoom() * viewportHeight / (workspaceChildrenRect.bottom())
        
            if widthRatio > heightRatio:
                ratio = heightRatio
            else:
                ratio = widthRatio
        
            self.setZoom(math.floor(ratio))
    
    def updateContent(self):
        """ Called after file is loaded.
        
        Meant to update to Tab content.
        """
        raise NotImplementedError

    def refresh(self):
        """ Reloads file content and refreshes tab.
        
        May be implemented by inheriting controllers.
        """
        statusMessage = self.plugin().application().startWorking("Reopening file")
        self._fileModifcationTimestamp = os.path.getmtime(self._filename)
        self.readFile(self._filename)
        self.updateContent()
        self.plugin().application().stopWorking(statusMessage)

    def zoomDialog(self):
        if hasattr(QInputDialog, "getInteger"):
            # Qt 4.3
            (zoom, ok) = QInputDialog.getInteger(self.tab(), "Zoom...", "Input zoom factor in percent:", self.zoom(), 0)
        else:
            # Qt 4.5
            (zoom, ok) = QInputDialog.getInt(self.tab(), "Zoom...", "Input zoom factor in percent:", self.zoom(), 0)
        if ok:
            self.setZoom(zoom)
            self._userZoomLevel = zoom

    def cancel(self):
        """ Cancel all operations in tab.
        
        This function is called when all current operations in tab shall be canceled.
        """
        pass

    def supportsUndo(self):
        """ Returns True if the this tab controller supports undo history.
        """
        return self._supportsUndo
    
    def enableUndo(self, enable=True):
        """ If enable is True this controller enables its undo function.
        
        For any tab controller that wants to use this feature, it needs to be made sure the corresponding UndoEvents
        for actions that should be undoable exists and are added by addUndoEvent().
        """
        self._supportsUndo = enable
            
    def undoEvents(self):
        """ Returns list of all registered UndoEvents.
        """
        return self._undoEvents
    
    def redoEvents(self):
        """ Returns list of all UndoEvents that have already been undone before.
        """
        return self._redoEvents
            
    def addUndoEvent(self, undoEvent):
        """ Adds event of type UndoEvent to this tab controller's list of undoable events.
        
        Undo can be invoked by calling undo().
        """
        if self._supportsUndo:
            if not isinstance(undoEvent, UndoEvent):
                logging.error("%s: Tried to add non-UndoEvent to list of undo events. Aborting..." % self.__class__.__name__)
                return
            self._redoEvents = []
            no_of_events = len(self._undoEvents)
            # try to combine similar events
            if no_of_events < 1 or not self._undoEvents[no_of_events -1].combine(undoEvent):
                self._undoEvents.append(undoEvent)
                if not self.isModified():
                    self.setLastSavedStateEvent(undoEvent)
                self.plugin().application().updateMenu()
        else:
            logging.warning("%s: Tried to add undo event, however undo functionality is not enabled. Aborting..." % self.__class__.__name__)    
        #self.dumpUndoEvents()
        
    def undo(self, numberOfEvents=1):
        """ Invokes undo of last stored UndoEvent (see addUndoEvent()), if undo functionality is enabled.
        """
        if not self._supportsUndo:
            logging.warning(self.__class__.__name__ + ": Tried to undo action, however undo functionality is not enabled. Aborting...")
            return
        logging.debug(self.__class__.__name__ +": undo("+ str(numberOfEvents) +")")
        #self.dumpUndoEvents()
        
        for i in range(0, numberOfEvents):
            if len(self._undoEvents) > 0:
                lastEvent = self._undoEvents.pop()
                lastEvent.undo()
                self._redoEvents.append(lastEvent)
                
                # set modification flag
                if i == (numberOfEvents -1):
                    if lastEvent.isLastSavedState():
                        self.setModified(False)
                    else:
                        self.setModified()
                self.plugin().application().updateMenu()
                
        #self.dumpUndoEvents()
        
    def redo(self, numberOfEvents=1):
        if not self._supportsUndo:
            logging.warning(self.__class__.__name__ + ": Tried to undo action, however undo functionality is not enabled. Aborting...")
            return
        logging.debug(self.__class__.__name__ +": redo("+ str(numberOfEvents) +")")
        #self.dumpUndoEvents()
        
        for i in range(0, numberOfEvents):
            if len(self._redoEvents) > 0:
                lastEvent = self._redoEvents.pop()
                lastEvent.redo()
                self._undoEvents.append(lastEvent)
                
                # set modification flag
                if i == (numberOfEvents -1):
                    undo_events_count = len(self._redoEvents)
                    if undo_events_count > 0:
                        if self._redoEvents[undo_events_count-1].isLastSavedState():
                            self.setModified(False)
                        else:
                            self.setModified()
                    else:
                        # if there are no more redo events
                        # and there is no event with last saved state flag set to True
                        # no action was performed since the last save
                        # and thus modification flag should be False
                        modified = False
                        for event in self._undoEvents + self._redoEvents:
                            if event.isLastSavedState():
                                modified = True
                                break
                        self.setModified(modified)
                self.plugin().application().updateMenu()
        
    def setLastSavedStateEvent(self, undoEvent):
        """ Sets last saved state flag of given UndoEvent to True and to False for all other events.
        """
        for current_event in self._undoEvents + self._redoEvents:
            if current_event == undoEvent:
                current_event.setLastSavedState(True)
            else:
                current_event.setLastSavedState(False)

    def dumpUndoEvents(self):
        for event in self._undoEvents:
            event.dump("undo")
        for event in self._redoEvents:
            event.dump("redo")
