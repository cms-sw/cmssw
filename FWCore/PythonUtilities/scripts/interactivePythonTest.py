#!/usr/bin/env python

if __name__ == '__main__':

    print "python -i `which interactivePythonTest.py` "
    
    #############################################
    ## Load and save command line history when ##
    ## running interactively.                  ##
    #############################################
    import os, readline
    import atexit
    historyPath = os.path.expanduser("~/.pyhistory")


    def save_history(historyPath=historyPath):
        import readline
        readline.write_history_file(historyPath)
        if os.path.exists(historyPath):
            readline.read_history_file(historyPath)


    atexit.register(save_history)
    readline.parse_and_bind("set show-all-if-ambiguous on")
    readline.parse_and_bind("tab: complete")
    if os.path.exists (historyPath) :
        readline.read_history_file(historyPath)
        readline.set_history_length(-1)


