from PyQt4.QtGui import *
from PyQt4.QtCore import *

class RunThread(QThread):
    """ Runs a command with attributes in a new thread.
    
        One can check if the thread is still running using isRunning().
        The return value of the command can be accessed using returnValue.
    """
    def __init__(self, command, *attr):
        QThread.__init__(self, None)
        self._command = command
        self._attr = attr
        self.returnValue = None
        self.start()
        
    def run(self):
        if len(self._attr) == 0:
            self.returnValue = self._command.__call__()
        elif len(self._attr) == 1:
            self.returnValue = self._command.__call__(self._attr[0])
        elif len(self._attr) == 2:
            self.returnValue = self._command.__call__(self._attr[1])
        else:
            raise NonImplementedError

class ThreadChain(QThread):
    """ Holds a list of commands that shall be executed in one Thread in a chain.

        Start the ThreadChain using start().
        One can check if the thread is still running using isRunning().
        When all commands are executed a signal "finishedThreadChain" will be emitted.
    """
    def __init__(self):
        QThread.__init__(self, None)
        self._commands = []
        
    def addCommand(self, command, attr=None):
        self._commands += [(command, attr)]
        
    def run(self):
        self._results = []
        while self._commands != []:
            command, attr = self._commands.pop()
            if attr != None:
                self._results += [command.__call__(attr)]
            else:
                self._results += [command.__call__()]
        self.emit(SIGNAL('finishedThreadChain'), self._results)
