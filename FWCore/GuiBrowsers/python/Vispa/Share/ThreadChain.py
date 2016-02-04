from PyQt4.QtCore import QThread,SIGNAL

import logging

class ThreadChain(QThread):
    """ Holds a list of commands that shall be executed in one Thread in a chain.

        The chain can run in two modes: Eighter it receives a command with optional attributes on construction. The return value can be accessed by calling returnValue() without arguments.
        In the second mode the constructor does not receive any arguments and commands are passed to the chain with addComand().
        This function returns an id unique for the command making the return value of the command available through retrunValue(id).
        Start the ThreadChain using start().
        
        One can check if the thread is still running using isRunning().
        When all commands are executed a signal "finishedThreadChain" will be emitted.
    """
    NO_THREADS_FLAG=False
    def __init__(self, command=None, *attr):
        QThread.__init__(self, None)
        self._commandTuples = []
        self._commandCounter = -1
        self._returnValues = {}
        if command:
            self.addCommand(command, *attr)
            self.start()
        
    def addCommand(self, command, *attr):
        """ Adds a command to this ThreadChain 
        
        and returns an id which is required to obtain the return value of this command.
        
        *attr is a optional tuple of arguments which will be passed to the command on execution.
        """
        self._commandCounter += 1
        id = self._commandCounter
        self._commandTuples += [(id, command, attr)]
        if self.NO_THREADS_FLAG:
            self._returnValues[id] = command.__call__(*attr)
            self.emit(SIGNAL('finishedThreadChain'), self._returnValues.values())
            return
        
    def clearReturnValues(self):
        self._returnValues.clear()
        
    def clearReturnValue(self, command):
        if command in self._returnValues.keys():
            self._returnValues.pop(command)
            return True
        return False
    
    def returnValue(self, id=None):
        """ Returns return value of command with given id.
        
        The id is returned by addCommand().
        If id is None the return value of the last command will be returned.
        """
        if id in self._returnValues.keys():
            return self._returnValues[id]
        valueLength = len(self._returnValues)
        if valueLength == 0:
            return []
            # TODO: maybe raise exception to distinguish from None return value?
        return self._returnValues[self._returnValues.keys()[valueLength-1]]
        
    def start(self):
        if self.NO_THREADS_FLAG:
            return
        if not self.isRunning():
            QThread.start(self)
        
    def run(self):
        if self.NO_THREADS_FLAG:
            return
        while self._commandTuples != []:
            id, command, attr = self._commandTuples.pop(0)
            self._returnValues[id] = command.__call__(*attr)
            
        # signal contains list of return values 
        # for compatibility with old implementation
        self.emit(SIGNAL('finishedThreadChain'), self._returnValues.values())
