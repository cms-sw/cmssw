import sys
import os
import logging

import Vispa.__init__

def setBaseDirectory(dir):
    global baseDirectory, mainDirectory, pluginDirectory, docDirectory
    baseDirectory = dir
    logging.debug(__name__ +': baseDirectory - '+baseDirectory)
    mainDirectory = os.path.join(baseDirectory, "Vispa/Main")
    logging.debug(__name__ +': mainDirectory - '+mainDirectory)
    pluginDirectory = os.path.join(baseDirectory, "Vispa/Plugins")
    logging.debug(__name__ +': pluginDirectory - '+pluginDirectory)
    docDirectory = os.path.join(baseDirectory, "doc/pxldoc")
    logging.debug(__name__ +': docDirectory - '+docDirectory)

def setHomeDirectory(dir):
    global homeDirectory, preferencesDirectory, iniFileName, logDirectory
    homeDirectory = dir
    logging.debug(__name__ +': homeDirectory - '+homeDirectory)
    preferencesDirectory = os.path.abspath(os.path.join(homeDirectory,".vispa"))
    logging.debug(__name__ +': preferencesDirectory - '+preferencesDirectory)
    iniFileName = os.path.abspath(os.path.join(preferencesDirectory,"vispa.ini"))
    logging.debug(__name__ +': iniFileName - '+iniFileName)
    logDirectory = os.path.abspath(preferencesDirectory)
    logging.debug(__name__ +': logDirectory - '+logDirectory)

def setWebsiteUrl(url):
    global websiteUrl
    websiteUrl=url

setBaseDirectory(os.path.abspath(os.path.dirname(Vispa.__path__[0])))
setHomeDirectory(os.path.expanduser("~"))
setWebsiteUrl("http://vispa.sourceforge.net")

applicationName=os.path.splitext(os.path.basename(sys.argv[0]))[0]
