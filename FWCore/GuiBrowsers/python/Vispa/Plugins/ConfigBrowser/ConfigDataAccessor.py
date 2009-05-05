import sys
import os.path
import logging
import re

from Vispa.Main.BasicDataAccessor import BasicDataAccessor
from Vispa.Main.RelativeDataAccessor import RelativeDataAccessor
from Vispa.Main.Exceptions import PluginIgnoredException
from Vispa.Main.Exceptions import exception_traceback

try:
    import ParameterSet_patch
    import FWCore.ParameterSet.SequenceTypes as sqt
    import FWCore.ParameterSet.Config as cms
    import FWCore.ParameterSet.Modules as mod
    import FWCore.ParameterSet.Types as typ
except Exception:
    raise PluginIgnoredException("cannot import CMSSW: " + exception_traceback())
    pass

imported_configs = {}
file_dict = {}

class ConfigFolder(object):
    def __init__(self, label, parent=None, parameters=None):
        self._label = label
        self._configChildren = []
        self._parameters = {}
        if parent != None:
            parent._configChildren += [self]
        if parameters != None:
            self._parameters = parameters
    def label_(self):
        return self._label
    def parameters_(self):
        return self._parameters
    def _configChildren(self):
        return self._configChildren
    
class ConfigDataAccessor(BasicDataAccessor, RelativeDataAccessor):
    def __init__(self):
        logging.debug(__name__ + ": __init__")

        self._allObjects = []
        self._connections = []
        self._topLevelObjects = []
        self._file = None
        self._configName = ""
        self._isReplaceConfig = False
        self._cancelOperationsFlag = False
    
    def cancelOperations(self):
        self._cancelOperationsFlag = True
    
    def isReplaceConfig(self):
        return self._isReplaceConfig 
        
    def setIsReplaceConfig(self):
        self._isReplaceConfig = True 
        
    def topLevelObjects(self):
        return self._topLevelObjects 

    def _readRecursive(self, mother, pth):
        """ Read cms objects recursively from path """
        entry = None
        if isinstance(pth, (cms.Path, cms.Sequence, cms.Source, mod._Module, cms.Service, cms.ESSource, cms.ESProducer, cms.ESPrefer, cms.PSet)):
            entry = pth
            self._allObjects += [pth]
            if mother != None:
                if hasattr(mother, "_configChildren"):
                    if not pth in mother._configChildren:
                        mother._configChildren += [pth]
                else:
                    mother._configChildren = [pth]
            else:
                self._topLevelObjects += [pth]
        next_mother = entry
        if entry == None:
            next_mother = mother
        if isinstance(pth, list):
            for i in pth:
                self._readRecursive(next_mother, i)
        for i in dir(pth):
            o = getattr(pth, i)
            if isinstance(o, sqt._Sequenceable):
                self._readRecursive(next_mother, o)
 
    def _readPaths(self, path_list, mother=None):
        """ Read objects from list of paths """
        for path in path_list:
            self._readRecursive(mother, path)

    def readConnections(self, objects):
        """ Read connection between objects """
        for entry1 in objects:
            if self._cancelOperationsFlag:
                break
            for key, value in self.inputTags(entry1):
                module = str(value).split(":")[0]
                product = ".".join(str(value).split(":")[1:])
                found = False
                for entry2 in objects:
                    if self._cancelOperationsFlag:
                        break
                    if module == self.label(entry2):
                        connection = (entry2, product, entry1, key)
                        found = True
                        if not connection in self._connections:
                            self._connections += [connection]
        ok = not self._cancelOperationsFlag
        self._cancelOperationsFlag = False
        return ok
    
    def connections(self):
        return self._connections

    def _sort_list(self, l):
        result = l[:]
        result.sort(lambda x, y: cmp(self.label(x).lower(), self.label(y).lower()))
        return result

    def open(self, filename=None):
        """ Open config file and read it.
        """
        logging.debug(__name__ + ": open")
        if filename != None:
            self._filename = filename
        global imported_configs
        (config_path, fileName) = os.path.split(str(self._filename))
        self._configName = os.path.splitext(fileName)[0]
        self._isReplaceConfig = False

# import input-config and make list of all imported configs
        for i in imported_configs.iterkeys():
            if i in sys.modules.keys():
                del sys.modules[i]
        sys.path.insert(0, config_path)
        common_imports = sys.modules.copy()

        import imp
        theFile = open(str(self._filename))
        self._file = imp.load_module(self._configName.replace(".", "_"), theFile, str(self._filename), ("py", "r", 1))
        theFile.close()
        
        imported_configs = sys.modules.copy()
        for i in common_imports.iterkeys():
            del imported_configs[i]
        
# make dictionary that connects every cms-object with the file in which it is defined
        for entry in dir(self.process()):
            file_dict[entry] = self._filename
        for j in imported_configs.itervalues():
          setj = set(dir(j))
          for entry in setj:
              if entry[0] != "_" and entry != "cms":
                source = 1
                for k in imported_configs.itervalues():
                    if hasattr(k, entry):
                      setk = set(dir(k))
                      if len(setk) < len(setj) and setk < setj:
                        source = 0
                if source == 1:
                    filen = self._filename
                    if hasattr(j, "__file__"):
                        filen = j.__file__
                    file_dict[entry] = filen

# collect all path/sequences/modules of the input-config in a list
        self._allObjects = []
        self._connections = []
        self._producedOutput = []
        self._topLevelObjects = []

        if self.process():
            parameters = {"name": self.process().process}
            process_folder = ConfigFolder("process", None, parameters)
            
            self._allObjects += [process_folder]
            self._topLevelObjects += [process_folder]

            folder_list = []
            folder_list += [("source", [self.process().source])]
            if self.process().schedule != None:
                folder_list += [("paths", self.process().schedule)]
            else:
                folder_list += [("paths", self.process().paths.itervalues())]
            folder_list += [("endpaths", self.process().endpaths.itervalues())]
            folder_list += [("services", self._sort_list(self.process().services.values()))]
            folder_list += [("psets", self._sort_list(self.process().psets.values()))]
            folder_list += [("vpsets", self._sort_list(self.process().vpsets.values()))]
            folder_list += [("essources", self._sort_list(self.process().es_sources.values()))]
            folder_list += [("esproducers", self._sort_list(self.process().es_producers.values()))]
            folder_list += [("esprefers", self._sort_list(self.process().es_prefers.values()))]
            for foldername, entry in folder_list:
                folder = ConfigFolder(foldername, process_folder)
                self._allObjects += [folder]
                self._readPaths(entry, folder)
        
            self._readHeaderInfo()
            if not self._isReplaceConfig:
                self.process().resetModified()
        else:
            path_list = []
            for entry in dir(self._file):
                if entry[0] != "_" and entry != "cms" and hasattr(getattr(self._file, entry), "label_"):
                    getattr(self._file, entry).setLabel(entry)
                    text = os.path.splitext(os.path.basename(file_dict[getattr(self._file, entry).label_()]))[0]
                    if text == self._configName:
                        path_list += [getattr(self._file, entry)]
            self._readPaths(path_list)
        return True

    def process(self):
        if hasattr(self._file, "process"):
            return self._file.process
        return None
    
    def _readHeaderInfo(self):
        theFile = open(str(self._filename))
        foundHeaderPart1 = False
        foundHeaderPart2 = False
        lines = 10
        while theFile and not (foundHeaderPart1 and foundHeaderPart2) and lines > 0:
            line = theFile.readline()
            lines -= 1
            if "Generated by ConfigBrowser" in line:
                foundHeaderPart1 = True
            splitline = line.split()
            if foundHeaderPart1 and len(splitline) == 4 and splitline[0] == "from" and splitline[2] == "import":
                self._configName = splitline[1]
                self._isReplaceConfig = True
                foundHeaderPart2 = True
        theFile.close()

    def dumpPython(self):
        """ dump python configuration """
        logging.debug(__name__ + ": dumpPython")
        text = ""
        if self.process():
            text += self.process().dumpPython()
        return text

    def configName(self):
        return self._configName

    def dumpReplaceConfig(self):
        """ dump replace python configuration """
        logging.debug(__name__ + ": dumpReplaceConfig")
        text = ""
        if self.process():
            text += "##########  Generated by ConfigBrowser  ##########\n"
            text += "from " + self._configName + " import *\n"
            text += "if hasattr(process,\"resetModified\"):\n"
            text += "    process.resetModified()\n"
            text += "##################################################\n\n"
            text += self.process().dumpModified()
        return text
        
    def label(self, object):
        """ Get label of an object """
        text = ""
        if hasattr(object, "label_"):
            text = object.label_()
        if text == "":
            if hasattr(object, "type_"):
                text = object.type_()
        return text

    def children(self, object):
        """ Get children of an object """
        if hasattr(object, "_configChildren"):
            return tuple(object._configChildren)
        else:
            return ()

    def nonSequenceChildren(self, object):
        objects = []
        if len(self.children(object)) == 0:
            objects = [object]
        else:
            for o in self.children(object):
                if len(self.children(o)) == 0:
                    objects += [o]
                else:
                    objects += [child for child in self.allChildren(o) if len(self.children(child)) == 0]
        return tuple(objects)
                
    def motherRelations(self, object):
        """ Get motherRelations of an object """
        objects = []
        for connection in self._connections:
            if connection[2] == object:
                objects += [connection[0]]
        return tuple(objects)

    def daughterRelations(self, object):
        """ Get daughterRelations of an object """
        objects = []
        for connection in self._connections:
            if connection[0] == object:
                objects += [connection[2]]
        return tuple(objects)

    def type(self, object):
        """ Get type of an object """
        text = ""
        if isinstance(object, cms.Path):
            text = "Path"
        elif isinstance(object, cms.Sequence):
            text = "Sequence"
        elif isinstance(object, cms.Source):
            text = "Source"
        elif isinstance(object, mod.EDProducer):
            text = "EDProducer"
        elif isinstance(object, mod.EDFilter):
            text = "EDFilter"
        elif isinstance(object, mod.EDAnalyzer):
            text = "EDAnalyzer"
        elif isinstance(object, mod.OutputModule):
            text = "OutputModule"
        elif isinstance(object, mod._Module):
            text = "Module"
        elif isinstance(object, cms.Service):
            text = "Service"
        elif isinstance(object, cms.ESSource):
            text = "ESSource"
        elif isinstance(object, cms.ESProducer):
            text = "ESProducer"
        elif isinstance(object, cms.ESPrefer):
            text = "ESPrefer"
        elif isinstance(object, cms.PSet):
            text = "PSet"
        return text

    def classname(self, object):
        """ Get classname of an object """
        text = ""
        if hasattr(object, "type_"):
            text = object.type_()
        return text

    def fullFilename(self, object):
        """ Get full filename """
        text = ""
#        if hasattr(object,"_filename"):
#            text=object._filename
        if text == "" or text.find("FWCore/ParameterSet") >= 0 or text.find("/build/") >= 0:
            if self.label(object) in file_dict:
                text = file_dict[self.label(object)]
        root = os.path.splitext(text)[0]
        if root != "":
            text = root + ".py"
        return text

    def lineNumber(self, object):
        """ Get linenumber """
        text = ""
        if hasattr(object, "_filename"):
            if object._filename.find("FWCore/ParameterSet") < 0 and object._filename.find("ConfigBrowser") < 0:
                if hasattr(object, "_lineNumber"):
                    text = str(object._lineNumber)
        return text

    def filename(self, object):
        """ Get filename """
        text = os.path.splitext(os.path.basename(self.fullFilename(object)))[0]
        return text
        
    def pypackage(self,object):
      match = re.match(r'(?:^|.*?/)([A-Za-z0-9_]*)/([A-Za-z0-9_]*)/(?:test|python)/((?:[A-Za-z0-9_]*/)*)([A-Za-z0-9_]*)\.py$',self.fullFilename(object))
      if match:
        return '%s.%s.%s%s' % (match.group(1),match.group(2),match.group(3).replace('/','.'),match.group(4))
      else:
        return ''

    def pypath(self,object):
      match = re.match(r'(?:^|.*?/)([A-Za-z0-9_]*/[A-Za-z0-9_]*/(?:test|python)/(?:[A-Za-z0-9_]*/)*[A-Za-z0-9_]*\.py)$',self.fullFilename(object))
      if match:
        return match.group(1)
      else:
        return ''

    def package(self, object):
        """ Get Package of an object file """
        shortdirname = os.path.dirname(self.fullFilename(object)).split('python/')
        text = ""
        if len(shortdirname) > 1:
            text = shortdirname[1]
        return text

    def parameters(self, object):
        """ Get parameters of an object """
        this_parameters = []
        if hasattr(object, "parameters_"):
            this_parameters = object.parameters_().items()
        elif hasattr(object, "_seq"):
            this_parameters = [('sequence', object._seq.dumpSequencePython())]
        if hasattr(object, "tarlabel_"):
            this_parameters += [('tarlabel', object.tarlabel_())]
        return this_parameters

    def _addInputTag(self, value, this_key, this_inputtags):
        """ Add alls inputtags of value to a list """
        if isinstance(value, list):
            for i in value:
                self._addInputTag(i, this_key, this_inputtags)
        if hasattr(value, "parameters_"):
            this_inputtags += self._readInputTagsRecursive(value.parameters_().items(), this_key)
        if isinstance(value, cms.InputTag):
            pythonValue = value.value()
            this_inputtags += [(str(this_key), value.value())]

    def _readInputTagsRecursive(self, this_parameters, start_key=""):
        """ Make list of inputtags from parameter dict """
        this_inputtags = []
        for key, value in this_parameters:
            this_key = start_key
            if this_key != "":
                this_key += "."
            this_key += key
            self._addInputTag(value, this_key, this_inputtags)
        return this_inputtags

    def inputTags(self, object):
        """ Make list of inputtags from parameter dict """
        return self._readInputTagsRecursive(self.parameters(object))

    def uses(self, object):
        """ Get list of all config objects that are used as input """
        uses = []
        for key, value in self.inputTags(object):
            module = str(value).split(":")[0]
            product = ".".join(str(value).split(":")[1:])
            if module not in uses:
                uses += [module]
        return uses
    
    def foundIn(self, object):
        """ Make list of all mother sequences """
        foundin = []
        for entry in self._allObjects:
            for daughter in self.children(entry):
                if self.label(object) == self.label(daughter) and len(self.children(entry)) > 0 and not self.label(entry) in foundin:
                    foundin += [self.label(entry)]
        return foundin

    def usedBy(self, object):
        """ Find config objects that use this as input """
        usedby = []
        for entry in self._allObjects:
            for uses in self.uses(entry):
                if self.label(object) == uses and not self.label(entry) in usedby:
                    usedby += [self.label(entry)]
        return usedby

    def recursePSetProperties(self, name, object, readonly=None):
        logging.debug(__name__ + ": recursePSetProperties: " + name)
        properties = []
        if name != "" and not isinstance(object, typ.PSet):
            try:
                if isinstance(object, cms.InputTag):
                    properties += [("String", name, "\"" + str(object.value()) + "\"", readonly)]
                elif hasattr(object, "pythonValue"):
                    properties += [("String", name, str(object.pythonValue()), readonly)]
                elif hasattr(object, "value"):
                    properties += [("Text", name, str(object.value()), readonly)]
                else:
                    properties += [("Text", name, str(object), readonly)]
            except Exception:
                logging.error(__name__ + ": " + exception_traceback())
        
        if isinstance(object, ConfigFolder):
            readonly = "readonly"
        
        params = self.parameters(object)[:]
        params.sort(lambda x, y: cmp(x[0].lower(), y[0].lower()))
        for key, value in params:
            keyname = name
            if name != "":
                keyname += "."
            keyname += key
            properties += self.recursePSetProperties(keyname, value, readonly)
        return properties
        
    def properties(self, object):
        """ Make list of all properties """
        properties = []
        properties += [("Category", "Object info", "")]
        if self.label(object) != "":
            properties += [("String", "label", self.label(object), "readonly")]
        if self.type(object) != "":
            text = self.type(object)
            if self.classname(object) != "":
                text += " <" + self.classname(object) + ">"
            properties += [("String", "type", text, "readonly")]
        if self.filename(object) != "":
            text = self.filename(object)
            if self.lineNumber(object) != "":
                text += " : " + self.lineNumber(object)
            properties += [("String", "file", text, "readonly")]
        if self.package(object) != "":
            properties += [("String", "package", self.package(object), "readonly")]
        if self.fullFilename(object) != "":
            properties += [("String", "full filename", self.fullFilename(object), "readonly")]
        if len(self.foundIn(object)) > 0:
            text = ""
            for entry in self.foundIn(object):
                if text != "":
                    text += ", "
                text += entry
            properties += [("String", "in sequence", text, "readonly")]
        if len(self.uses(object)) + len(self.usedBy(object)) > 0:
            properties += [("Category", "Connections", "")]
            if len(self.uses(object)) > 0:
                text = ""
                for label in self.uses(object):
                    if text != "":
                        text += ", "
                    text += label
                properties += [("Text", "uses", text, "readonly")]
            if len(self.usedBy(object)) > 0:
                text = ""
                usedby = []
                for entry in self.usedBy(object):
                    if text != "":
                        text += ", "
                    text += entry
                properties += [("Text", "used by", text, "readonly")]
        if len(self.parameters(object)) > 0:
            properties += [("Category", "Parameters", "")]
            properties += self.recursePSetProperties("", object)
        return tuple(properties)
    
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

    def setProperty(self, object, name, value):
        """ Sets a property with given name to value.
        """
        try:
            exec "object." + name + "=" + value
        except Exception:
            return False
        return True

    def inputEventContent(self):
        content = []
        allLabels = [self.label(object) for object in self._allObjects]
        for entry in self._allObjects:
            for uses in self.uses(entry):
                if not uses in allLabels and not uses in content:
                    content += [uses]
        return content

    def outputEventContent(self):
        content = [self.label(object) for object in self._allObjects\
                 if self.type(object) in ["EDProducer", "EDFilter", "EDAnalyzer"]]
        return content
    
    def inputCommands(self):
        inputModules = [object for object in self._allObjects\
                        if self.type(object) == "Source"]
        if len(inputModules) > 0 and hasattr(inputModules[0], "inputCommands"):
            return inputModules[0].inputCommands
        else:
            return []

    def outputCommands(self):
        outputModules = [object for object in self._allObjects\
                        if self.type(object) == "OutputModule"]
        if len(outputModules) > 0 and hasattr(outputModules[0], "inputCommands"):
            return outputModules[0].outputCommands
        else:
            return []

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
