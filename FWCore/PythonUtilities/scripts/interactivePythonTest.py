#!/usr/bin/env python3
from __future__ import print_function
import sys
import os
import readline
import atexit

import ctypes

def interactive_inspect_mode():
    # http://stackoverflow.com/questions/640389/tell-whether-python-is-in-i-mode
    flagPtr = ctypes.cast(ctypes.pythonapi.Py_InteractiveFlag, 
                         ctypes.POINTER(ctypes.c_int))
    return flagPtr.contents.value > 0 or bool(os.environ.get("PYTHONINSPECT",False))


if __name__ == '__main__':

    #############################################
    ## Load and save command line history when ##
    ## running interactively.                  ##
    #############################################
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

    if not interactive_inspect_mode():
        print("python -i `which interactivePythonTest.py` ")


