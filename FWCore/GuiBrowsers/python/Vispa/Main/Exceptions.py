import sys
import traceback

class NoCurrentTabControllerException(Exception):
    """ This exception is raised if a function tries to access the Application's current tab controller when there is no currentTab() in tabWidget().
    """
    pass

class PluginIgnoredException(Exception):
    """ This exception is raised if a plugin cannot be loaded and shall be ignored.
    """
    pass

class PluginNotLoadedException(ImportError):
    """ This exception is raised if a plugin cannot be loaded and shall raise a warning.
    """
    pass

def exception_traceback():
    ty,va,tb=sys.exc_info()
    return "".join(traceback.format_exception(ty,va,tb))
