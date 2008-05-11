#!/usr/bin/env python

#################################################
#
# DQMfromDBSgui.py
#
# v1.4 Beta
#
# by Jeff Temple (jtemple@fnal.gov)
#
# 11 May 2008
#
# v1.3 updates -- separate code into subpackages
#      introduce separate file, dataset substrings
#      reposition daughter windows relative to parent
#
# v1.4 updates -- DQMfromDBSgui.py now inherits from base
#   class in pyDBSguiBaseClass.py
#   Other DQM tools can also inherit from this base class
#
# DQMfromDBSgui.py:
# GUI to automatically grab new runs from DBS
# and run HCAL DQM on any newly-found runs
#
#################################################

import sys
import os # used for sending user commands
base=os.popen2("echo $CMSSW_BASE")
base=base[1].read()
if len(base)<2:
    print "No $CMSSW_BASE directory can be found."
    print "Are you sure you've set up your CMSSW release area?"
    sys.exit()


try:
    from Tkinter import *
except:
    print "Could not import Tkinter module!"
    print "Cannot display GUI!"
    print "(If you are running from FNAL, this is a known problem --\n"
    print " Tkinter isn't available in the SL4 release of python for some reason."
    print "This is being investigated.)"
    sys.exit()

import tkMessageBox # for displaying warning messages

import thread # used for automated checking; may not be necessary
import time # use to determine when to check for files
import cPickle # stores file information
import python_dbs # use for "sendmessage" function to get info from DBS
import string # DBS output parsing
import helpfunctions  # displays text in new window


from pyDBSRunClass import DBSRun  # Gets class that holds information as to whether a given run has been processed through DQM

from pydbsAccessor import dbsAccessor  # sends queries to DBS
from pyDBSguiBaseClass import DQMDBSgui # get main gui package


###########################################################

if __name__=="__main__":

    debug=False

    # Use option parsing to enable/disable debugging from the command line
    try:
        from optparse import OptionParser
        parser=OptionParser()
        # If "-d" option used, set debug variable true
        parser.add_option("-d","--debug",
                          dest="debug",
                          action="store_true",
                          help = "Enable debugging when GUI runs",
                          default=False)

        (options,args)=parser.parse_args(sys.argv)
        debug=options.debug
    except:
        print "Dang, optionParsing didn't work"
        print "Starting GUI anyway"


    # Now run the GUI
    mygui=DQMDBSgui(debug=debug)  # set up gui
    mygui.DrawGUI()
    mygui.root.mainloop() # run main loop
