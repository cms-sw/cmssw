class AbstractTab(object):
    """ Abstract class for tabs to which a TabController can be attached.
    """
    def __init__(self):
        self._controller = None
        self._tabWidget = None
        self._mainWindow = None
    
    def setController(self, controller):
        """ Attaches a controller to the Tab,
        
        The controller() variable of the Tab and the tab() variable of the controller are set.
        """
        self._controller = controller
        self._controller.setTab(self)
        
    def controller(self):
        return self._controller
    
    def setTabWidget(self, widget):
        """ Sets the tabWidget variable, which is returned by tabWidget().
        """
        self._tabWidget = widget
        
    def tabWidget(self):
        """ Returns the tabWidget set by setTabWidget().
        
        Important for updating the tab's label etc.
        """
        return self._tabWidget
    
    def setMainWindow(self, main):
        """Sets the mainWindow variable, which is returned by mainWindow().
        """
        self._mainWindow = main
        
    def mainWindow(self):
        """Returns the main window widget.
        
        Especially for dialog boxes.
        """
        return self._mainWindow
    