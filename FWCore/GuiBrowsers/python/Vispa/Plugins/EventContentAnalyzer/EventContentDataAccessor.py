import sys
import logging
import os.path

from PyQt4.QtCore import QCoreApplication

from Vispa.Main.Thread import *
from Vispa.Main.BasicDataAccessor import *
from Vispa.Plugins.ConfigBrowser.ConfigDataAccessor import *
from Vispa.Main.Exceptions import PluginIgnoredException
from Vispa.Main.Exceptions import exception_traceback

try:
    import PhysicsTools.PythonAnalysis as cmstools
    import ROOT
except Exception:
    raise PluginIgnoredException("cannot import CMSSW: " + exception_traceback())
    pass

class EventContentDataAccessor(BasicDataAccessor):
    def __init__(self):
        self._eventContents = []
        self._configAccessors = {}
        
    def children(self, parent):
        return []
        
    def label(self, object):
        return "_".join(object)

    def eventContentsList(self):
        """ return top level objects from file, e.g. the event.
        """
        return self._eventContents
    
    def topLevelObjects(self):
        objects = []
        for name, content, input in self._eventContents:
            objects += list(content)
        return objects

    def openConfigAccessor(self, filename):
        if filename in self._configAccessors.keys():
            accessor = self._configAccessors[filename]
            return accessor
        accessor = ConfigDataAccessor()
        thread = RunThread(accessor.open, filename)
        while thread.isRunning():
            QCoreApplication.instance().processEvents()
        if not thread.returnValue:
            logging.error("Could not open config file: " + str(filename))
            return None
        if not accessor.process():
            logging.error("Config file does not contain process: " + str(filename))
            return None
        self._configAccessors[filename] = accessor
        return accessor

    def addContents(self,content,content2):
        for object in content2:
            if not self.inContent(object,content):
                content+=[object]

    def addConfigFile(self, filename):
        accessor = self.openConfigAccessor(filename)
        if not accessor:
            return
        name = "Input: " + accessor.configName()
        self._eventContents += [(name, accessor.inputEventContent(), True)]
        output_content=[]
        if len(self._eventContents)>1:
            self.addContents(output_content,self._eventContents[-2][1])
        self.applyCommands(output_content,accessor.inputCommands())
        self.addContents(output_content,accessor.outputEventContent())
        self.applyCommands(output_content,accessor.outputCommands())
        name = "Output: " + accessor.configName()
        self._eventContents += [(name, output_content, True)]

    def properties(self, object):
        """ Make list of all properties """
        properties = []
        properties += [("Category", "Object info", "")]
        properties += [("Text", "Label", object)]
        return properties

    def propertyValue(self, object, name):
        """ Returns value of property with given name.
        """
        propertiesDict = {}
        for p in self.properties(object):
            propertiesDict[p[1]] = p[2]
        if name in propertiesDict.keys():
            return propertiesDict[name]
        else:
            return None

    def branchType(self, branch):
        type = cmstools.ROOT.branchToClass(branch).GetName()
        if "edm::Wrapper" in type:
            type = type.replace("edm::Wrapper<", "").rstrip(">")
        return type.strip(" ")

    def addRootFile(self, filename):
        ROOT.gSystem.Load("libFWCoreFWLite.so")
        ROOT.AutoLibraryLoader.enable()
        events = cmstools.EventTree(filename)
        listOfBranches = events._tree.GetListOfBranches()
        content = []
        for branch in listOfBranches:
            name = branch.GetName()
            if not "EventAux" in name:
                type = self.branchType(branch)
                module = name.split("_")[1]
                product = name.split("_")[2]
                process = name.split("_")[3]
                cpp = events.cppCode(name)
                content += [(type,module,product,process)]
        name = os.path.splitext(os.path.basename(filename))[0]
        self._eventContents += [(name, content, False)]

    def addTextFile(self, filename):
        file = open(filename)
        content = []
        for line in file.readlines():
            linecontent=[l.strip(" \n") for l in line.split("\"")]
            content += [(linecontent[0],linecontent[1],linecontent[3],linecontent[5])]
        name = os.path.splitext(os.path.basename(filename))[0]
        self._eventContents += [(name, content, False)]

    def compareEntry(self, entry1, entry2):
        result=True
        for i in range(4):
            result=result and (entry1[i]==entry2[i] or entry1[i]=="*" or entry2[i]=="*")
        return result

    def inContent(self, entry, content):
        return True in [self.compareEntry(entry,c) for c in content]

    def applyCommands(self, content, outputCommands):
        keep = {}
        for object in content:
            keep[object] = True
        for o in outputCommands:
            command, filter = o.split(" ")
            if len(filter.split("_")) > 1:
                module = filter.split("_")[1]
            else:
                module = filter
            for object in content:
                if "*" in module:
                    match = module.strip("*") in object
                else:
                    match = module == object
                if match:
                    keep[object] = command == "keep"
        return [object for object in content if keep[object]]
