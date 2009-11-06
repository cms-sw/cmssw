import sys
import os.path
import logging
import re

from Vispa.Share.BasicDataAccessor import BasicDataAccessor
from Vispa.Share.RelativeDataAccessor import RelativeDataAccessor
from Vispa.Main.Exceptions import PluginIgnoredException,exception_traceback

import FWCore.ParameterSet.SequenceTypes as sqt
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Modules as mod
import FWCore.ParameterSet.Types as typ
import FWCore.GuiBrowsers.ParameterSet_patch 

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

        self._file = None
        self._filename=""
        self._isReplaceConfig = False
        self._cancelOperationsFlag = False
        self._initLists()
    
    def _initLists(self):
        self._allObjects = []
        self._connections = []
        self._topLevelObjects = []
        self._inputTagsDict = {}
        self._foundInDict = {}
        self._usesDict = {}
        self._usedByDict = {}
        self._motherRelationsDict = {}
        self._daughterRelationsDict = {}

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
                            if not entry1 in self._motherRelationsDict.keys():
                                self._motherRelationsDict[entry1]=[]
                            self._motherRelationsDict[entry1]+=[entry2]
                            if not entry2 in self._daughterRelationsDict.keys():
                                self._daughterRelationsDict[entry2]=[]
                            self._daughterRelationsDict[entry2]+=[entry1]
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
            self._filename = str(filename)
        global imported_configs
        self._isReplaceConfig = False

# import input-config and make list of all imported configs
        for i in imported_configs.iterkeys():
            if i in sys.modules.keys():
                del sys.modules[i]
        sys.path.insert(0, os.path.dirname(self._filename))
        common_imports = sys.modules.copy()

        import imp
        theFile = open(self._filename)
        self._file = imp.load_module(os.path.splitext(os.path.basename(self._filename))[0].replace(".", "_"), theFile, self._filename, ("py", "r", 1))
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
        if self.process():
            self.setProcess(self.process())
            self._readHeaderInfo()
            if not self._isReplaceConfig:
                self.process().resetHistory()
        else:
            self._initLists()
            path_list = []
            for entry in dir(self._file):
                if entry[0] != "_" and entry != "cms" and hasattr(getattr(self._file, entry), "label_"):
                    getattr(self._file, entry).setLabel(entry)
                    text = os.path.splitext(os.path.basename(file_dict[getattr(self._file, entry).label_()]))[0]
                    if text == os.path.splitext(os.path.basename(self._filename))[0]:
                        path_list += [getattr(self._file, entry)]
            self._readPaths(path_list)
        return True

    def setProcess(self,process):
        self._file.process=process
        self._initLists()
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

    def process(self):
        if hasattr(self._file, "process"):
            return self._file.process
        return None
    
    def _readHeaderInfo(self):
        theFile = open(self._filename)
        foundHeaderPart1 = False
        foundHeaderPart2 = False
        lines = 10
        dirname=os.path.dirname(self._filename)
        while theFile and not (foundHeaderPart1 and foundHeaderPart2) and lines > 0:
            line = theFile.readline()
            lines -= 1
            if "Generated by ConfigEditor" in line:
                foundHeaderPart1 = True
            splitline = line.split("'")
            if foundHeaderPart1 and len(splitline) == 3 and splitline[0] == "sys.path.append(os.path.abspath(os.path.expandvars(" and splitline[2] == ")))\n":
                dirname=splitline[1]
            splitline = line.split()
            if foundHeaderPart1 and len(splitline) == 4 and splitline[0] == "from" and splitline[2] == "import":
                self._filename = os.path.join(os.path.abspath(os.path.expandvars(dirname)),splitline[1])
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

    def configFile(self):
        return self._filename

    def label(self, object):
        """ Get label of an object """
        text = ""
        if hasattr(object, "label_"):
            text = str(object.label_())
        if text == "":
            if hasattr(object, "type_"):
                text = str(object.type_())
        return text

    def children(self, object):
        """ Get children of an object """
        if hasattr(object, "_configChildren"):
            return tuple(object._configChildren)
        else:
            return ()
        
    def isContainer(self, object):
        return isinstance(object, (ConfigFolder, list, cms.Path, cms.Sequence))

    def nonSequenceChildren(self, object):
        objects = []
        if not self.isContainer(object):
            objects = [object]
        else:
            for o in self.children(object):
                if not self.isContainer(o):
                    objects += [o]
                else:
                    objects += [child for child in self.allChildren(o) if len(self.children(child)) == 0]
        return tuple(objects)
                
    def motherRelations(self, object):
        """ Get motherRelations of an object """
        if object in self._motherRelationsDict.keys():
            return self._motherRelationsDict[object]
        else:
            return []

    def daughterRelations(self, object):
        """ Get daughterRelations of an object """
        if object in self._daughterRelationsDict.keys():
            return self._daughterRelationsDict[object]
        else:
            return []

    def type(self, object):
        """ Get type of an object """
        return object.__class__.__name__ 

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
            if object._filename.find("FWCore/ParameterSet") < 0 and object._filename.find("ConfigEditor") < 0:
                if hasattr(object, "_lineNumber"):
                    text = str(object._lineNumber)
        return text

    def filename(self, object):
        """ Get filename """
        text = os.path.splitext(os.path.basename(self.fullFilename(object)))[0]
        return text
        
    def pypackage(self,object):
      match_compiled = re.match(r'(?:^|.*?/)CMSSW[0-9_]*/python/((?:\w*/)*\w*)\.py$',self.fullFilename(object))
      if match_compiled:
        return match_compiled.group(1).replace('/','.')
      
      match_norm = re.match(r'(?:^|.*?/)(\w*)/(\w*)/(?:test|python)/((?:\w*/)*)(\w*)\.py$',self.fullFilename(object))
      if match_norm:
        return '%s.%s.%s%s' % (match_norm.group(1),match_norm.group(2),match_norm.group(3).replace('/','.'),match_norm.group(4))
      return ''

    def pypath(self,object):
      match_compiled = re.match(r'(?:^|.*?/)CMSSW[0-9_]*/python/((?:\w*/){2})((?:\w*/)*)(\w*\.py)$',self.fullFilename(object))
      if match_compiled:
        return '%spython/%s%s' % (match_compiled.group(1),match_compiled.group(2),match_compiled.group(3))
      match_norm = re.match(r'(?:^|.*?/)(\w*/\w*/(?:test|python)/(?:\w*/)*\w*\.py)$',self.fullFilename(object))
      if match_norm:
        return match_norm.group(1)
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
        if not object in self._inputTagsDict.keys():
             self._inputTagsDict[object]=self._readInputTagsRecursive(self.parameters(object))
        return self._inputTagsDict[object]

    def uses(self, object):
        """ Get list of all config objects that are used as input """
        if not object in self._usesDict.keys():
            uses = []
            for key, value in self.inputTags(object):
                module = str(value).split(":")[0]
                product = ".".join(str(value).split(":")[1:])
                if module not in uses:
                    uses += [module]
            self._usesDict[object]=uses
        return self._usesDict[object]
    
    def foundIn(self, object):
        """ Make list of all mother sequences """
        if not object in self._foundInDict.keys():
            foundin = []
            for entry in self._allObjects:
                for daughter in self.children(entry):
                    if self.label(object) == self.label(daughter) and len(self.children(entry)) > 0 and not self.label(entry) in foundin:
                        foundin += [self.label(entry)]
            self._foundInDict[object]=foundin
        return self._foundInDict[object]

    def usedBy(self, object):
        """ Find config objects that use this as input """
        if not object in self._usedByDict.keys():
            usedby = []
            for entry in self._allObjects:
                for uses in self.uses(entry):
                    if self.label(object) == uses and not self.label(entry) in usedby:
                        usedby += [self.label(entry)]
            self._usedByDict[object]=usedby
        return self._usedByDict[object]

    def recursePSetProperties(self, name, object, readonly=None):
        #logging.debug(__name__ + ": recursePSetProperties: " + name)
        properties = []
        if name != "" and not isinstance(object, typ.PSet):
            try:
                if isinstance(object, cms.InputTag):
                    properties += [("String", name, "\"" + str(object.value()) + "\"", None, readonly)]
                elif hasattr(object, "pythonValue"):
                    properties += [("String", name, str(object.pythonValue()), None, readonly)]
                elif hasattr(object, "value"):
                    properties += [("MultilineString", name, str(object.value()), None, readonly)]
                else:
                    properties += [("MultilineString", name, str(object), None, readonly)]
            except Exception:
                logging.error(__name__ + ": " + exception_traceback())
        
        if isinstance(object, ConfigFolder):
            readonly = True
        
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
        #logging.debug(__name__ + ": properties")
        properties = []
        properties += [("Category", "Object info", "")]
        if self.label(object) != "":
            properties += [("String", "label", self.label(object), None, True)]
        if self.type(object) != "":
            text = self.type(object)
            if self.classname(object) != "":
                text += " <" + self.classname(object) + ">"
            properties += [("String", "type", text, None, True)]
        if self.filename(object) != "":
            text = self.filename(object)
            if self.lineNumber(object) != "":
                text += " : " + self.lineNumber(object)
            properties += [("String", "file", text, None, True)]
        if self.package(object) != "":
            properties += [("String", "package", self.package(object), None, True)]
        if self.fullFilename(object) != "":
            properties += [("String", "full filename", self.fullFilename(object), None, True)]
        foundIn=self.foundIn(object)
        if len(foundIn) > 0:
            text = ""
            for entry in foundIn:
                if text != "":
                    text += ", "
                text += entry
            properties += [("String", "in sequence", text, None, True)]
        uses=self.uses(object)
        usedBy=self.usedBy(object)
        if len(uses) + len(usedBy) > 0:
            properties += [("Category", "Connections", "")]
            if len(uses) > 0:
                text = ""
                for label in uses:
                    if text != "":
                        text += ", "
                    text += label
                properties += [("MultilineString", "uses", text, None, True)]
            if len(usedBy) > 0:
                text = ""
                usedby = []
                for entry in usedBy:
                    if text != "":
                        text += ", "
                    text += entry
                properties += [("MultilineString", "used by", text, None, True)]
        if len(self.parameters(object)) > 0:
            properties += [("Category", "Parameters", "")]
            properties += self.recursePSetProperties("", object)
        return tuple(properties)
    
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
        for object in self._allObjects:
            for key, value in self.inputTags(object):
                elements=str(value).split(":")
                module = elements[0]
                if len(elements)>1 and elements[1]!="":
                    product = elements[1]
                else:
                    product = "*"
                if len(elements)>2 and elements[2]!="":
                    process = elements[2]
                else:
                    process = "*"
                if not module in allLabels and not ("*",module,product,process) in content:
                    content += [("*",module,product,process)]
        return content

    def outputEventContent(self):
        content = [("*",self.label(object),"*","*") for object in self._allObjects\
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
