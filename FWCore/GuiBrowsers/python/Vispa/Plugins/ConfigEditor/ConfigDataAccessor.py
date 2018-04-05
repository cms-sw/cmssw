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
        self._history=None
        self._cancelOperationsFlag = False
        self._initLists()
    
    def _initLists(self):
        self._allObjects = []
        self._scheduledObjects = []
        self._connections = {}
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
        if isinstance(pth, (cms.Path, cms.EndPath, cms.Sequence, cms.SequencePlaceholder, cms.Source, mod._Module, cms.Service, cms.ESSource, cms.ESProducer, cms.ESPrefer, cms.PSet, cms.VPSet)):
            entry = pth
            entry._configChildren=[]
            self._allObjects += [pth]
            if mother != None:
                if not pth in mother._configChildren:
                    mother._configChildren += [pth]
            else:
                self._topLevelObjects += [pth]
        next_mother = entry
        if entry == None:
            next_mother = mother
        if isinstance(pth, list):
            for i in pth:
                self._readRecursive(next_mother, i)
        if hasattr(sqt,"_SequenceCollection"):
            # since CMSSW_3_11_X
            if isinstance(pth, (sqt._ModuleSequenceType)):
              if isinstance(pth._seq, (sqt._SequenceCollection)):
                for o in pth._seq._collection:
                    self._readRecursive(next_mother, o)
              else:
                  self._readRecursive(next_mother, pth._seq)
            elif isinstance(pth, sqt._UnarySequenceOperator):
                self._readRecursive(next_mother, pth._operand)
        else:
            # for backwards-compatibility with CMSSW_3_10_X
            for i in dir(pth):
                o = getattr(pth, i)
                if isinstance(o, sqt._Sequenceable):
                    self._readRecursive(next_mother, o)
 
    def readConnections(self, objects,toNeighbors=False):
        """ Read connection between objects """
        connections={}
        checkedObjects = set()
        self._motherRelationsDict={}
        self._daughterRelationsDict={}
        if toNeighbors:
            compareObjectList=[]
            for obj in objects:
                compareObjectList+=[(obj,o) for o in self._allObjects]
                compareObjectList+=[(o,obj) for o in self._allObjects]
        else:
            compareObjectList=[(o1,o2) for o1 in objects for o2 in objects]
        for connection in compareObjectList:
            if self._cancelOperationsFlag:
                break
            try:
                if (not connection in checkedObjects) and (not self._connections.has_key(connection)):
                    checkedObjects.add(connection)
                    for key, value in self.inputTags(connection[1]):
                        s = str(value)
                        index = s.find(':')
                        if -1 != index:
                            module = s[:index]
                        else:
                            module = s
                        if module == self.label(connection[0]):
                            product = ".".join(str(value).split(":")[1:])
                            self._connections[connection]=(product, key)
                if self._connections.has_key(connection):
                    connections[connection]=self._connections[connection]
                    if not self._motherRelationsDict.has_key(connection[1]):
                        self._motherRelationsDict[connection[1]]=[]
                    self._motherRelationsDict[connection[1]]+=[connection[0]]
                    if not self._daughterRelationsDict.has_key(connection[0]):
                        self._daughterRelationsDict[connection[0]]=[]
                    self._daughterRelationsDict[connection[0]]+=[connection[1]]
            except TypeError:
                return {}
        return connections
    
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
        self._history=None

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
            self._history=self.process().dumpHistory()
            if not self._isReplaceConfig and hasattr(self.process(),"resetHistory"):
                self.process().resetHistory()
        else:
            self._initLists()
            for entry in dir(self._file):
                o=getattr(self._file, entry)
                if entry[0] != "_" and entry != "cms" and hasattr(o, "label_"):
                    getattr(self._file, entry).setLabel(entry)
                    text = os.path.splitext(os.path.basename(file_dict[o.label_()]))[0]
                    if text == os.path.splitext(os.path.basename(self._filename))[0] and not o in self._allObjects:
                        self._readRecursive(None, o)
        return True

    def _scheduleRecursive(self,object):
        if object in self._scheduledObjects:
	    return
        if self.isContainer(object):
	    for obj in reversed(self.children(object)):
	        self._scheduleRecursive(obj)
        else:
            self._scheduledObjects+=[object]
	    for used in self.motherRelations(object):
	        self._scheduleRecursive(used)

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
        folder_list += [("modules", self._sort_list(self.process().producers.values()+self.process().filters.values()+self.process().analyzers.values()))]
        folder_list += [("services", self._sort_list(self.process().services.values()))]
        folder_list += [("psets", self._sort_list(self.process().psets.values()))]
        folder_list += [("vpsets", self._sort_list(self.process().vpsets.values()))]
        folder_list += [("essources", self._sort_list(self.process().es_sources.values()))]
        folder_list += [("esproducers", self._sort_list(self.process().es_producers.values()))]
        folder_list += [("esprefers", self._sort_list(self.process().es_prefers.values()))]
        folders={}
	for foldername, entry in folder_list:
            folder = ConfigFolder(foldername, process_folder)
            self._allObjects += [folder]
	    folders[foldername]=folder
            for path in entry:
                self._readRecursive(folder, path)
	if True:
            print "Creating schedule...",
            self.readConnections(self.allChildren(folders["modules"]))
	    self._scheduleRecursive(folders["paths"])
	    self._scheduledObjects.reverse()
	    names = [l for t,l,p,pr in self.applyCommands(self.outputEventContent(),self.outputCommands())]
	    for obj in self.allChildren(folders["modules"]):
	       if str(obj) in names:
	           self._scheduledObjects+=[obj]
            scheduled_folder = ConfigFolder("scheduled", folders["paths"])
            self._allObjects += [scheduled_folder]
            folders["paths"]._configChildren.remove(scheduled_folder)
            folders["paths"]._configChildren.insert(0,scheduled_folder)
	    scheduled_folder._configChildren=self._scheduledObjects
	    print "done"
        else:
	    self._scheduledObjects=self._allObjects

    def process(self):
        if hasattr(self._file, "process"):
            return self._file.process
        return None
    
    def _readHeaderInfo(self):
        theFile = open(self._filename)
        foundHeaderPart1 = False
        foundHeaderPart2 = False
        lines = 10
        search_paths=[os.path.abspath(os.path.dirname(self._filename))]
        while theFile and not (foundHeaderPart1 and foundHeaderPart2) and lines > 0:
            line = theFile.readline()
            lines -= 1
            if "Generated by ConfigEditor" in line:
                foundHeaderPart1 = True
            splitline = line.split("'")
            if foundHeaderPart1 and len(splitline) == 5 and splitline[0] == "sys.path.append(os.path.abspath(os.path.expandvars(os.path.join(" and splitline[4] == "))))\n":
                search_paths+=[os.path.abspath(os.path.expandvars(os.path.join(splitline[1],splitline[3])))]
            splitline = line.split()
            if foundHeaderPart1 and len(splitline) == 4 and splitline[0] == "from" and splitline[2] == "import":
                for search_path in search_paths:
                    if os.path.exists(os.path.join(search_path,splitline[1]+".py")):
                        self._filename = os.path.join(search_path,splitline[1]+".py")
                        break
                self._isReplaceConfig = True
                foundHeaderPart2 = True
        theFile.close()

    def dumpPython(self):
        """ dump python configuration """
        logging.debug(__name__ + ": dumpPython")
        text = None
        if self.process():
            text = self.process().dumpPython()
        return text

    def history(self):
        """ configuration history """
        logging.debug(__name__ + ": history")
        return self._history

    def configFile(self):
        return self._filename

    def label(self, object):
        """ Get label of an object """
        text = ""
        if hasattr(object, "label_") and (not hasattr(object,"hasLabel_") or object.hasLabel_()):
            text = str(object.label_())
            if text:
                return text
        if text == "":
            if hasattr(object, "_name"):
                text = str(object._name)
        if text == "":
            if hasattr(object, "type_"):
                text = str(object.type_())
        if text == "":
            text = "NoLabel"
        return text

    def children(self, object):
        """ Get children of an object """
        if hasattr(object, "_configChildren"):
            return tuple(object._configChildren)
        else:
            return ()
        
    def isContainer(self, object):
        return isinstance(object, (ConfigFolder, list, cms.Path, cms.EndPath, cms.Sequence)) # cms.SequencePlaceholder assumed to be a module

    def nonSequenceChildren(self, object):
        objects=[]
        if self.isContainer(object):
            for o in self.allChildren(object):
                if not self.isContainer(o) and len(self.children(o)) == 0 and not o in objects:
                    objects += [o] 
        else:
            for o in self.motherRelations(object)+[object]+self.daughterRelations(object):
                if not o in objects:
                    objects += [o] 
        return objects
                
    def motherRelations(self, object):
        """ Get motherRelations of an object """
        if object in self._motherRelationsDict.keys():
	    try:
                return self._motherRelationsDict[object]
	    except TypeError:
	        return []
        else:
            return []

    def daughterRelations(self, object):
        """ Get daughterRelations of an object """
        if object in self._daughterRelationsDict.keys():
	    try:
                return self._daughterRelationsDict[object]
	    except TypeError:
	        return []
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
            else:
                text = self._filename
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
            if hasattr(object._seq,"dumpSequencePython"):
                this_parameters = [('sequence', object._seq.dumpSequencePython())]
            else:
                this_parameters = [('sequence', 'WARNING: object was removed from a sequence.')]
        if hasattr(object, "tarlabel_"):
            this_parameters += [('tarlabel', object.tarlabel_())]
        return this_parameters

    def _addInputTag(self, value, this_key, this_inputtags):
        """ Add alls inputtags of value to a list """
        if isinstance(value, cms.VInputTag):
            for i in range(len(value)):
                if type(value[i])==str:
                    self._addInputTag(cms.InputTag(value[i]), this_key+"["+str(i)+"]", this_inputtags)
                else:
                    self._addInputTag(value[i], this_key+"["+str(i)+"]", this_inputtags)
        elif isinstance(value, list):
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
        try:
            v = self._inputTagsDict.get(object)
            if v is None:
                v = self._readInputTagsRecursive(self.parameters(object))
                self._inputTagsDict[object]=v
        except TypeError:
            v = []
        return v

    def uses(self, object):
        """ Get list of all config objects that are used as input """
        if not object in self._usesDict.keys():
            uses = []
            for key, value in self.inputTags(object):
                module = str(value).split(":")[0]
                product = ".".join(str(value).split(":")[1:])
                if module not in uses:
                    uses += [module]
	    try:
                self._usesDict[object]=uses
	    except TypeError:
	        return []
        return self._usesDict[object]
    
    def foundIn(self, object):
        """ Make list of all mother sequences """
        if not object in self._foundInDict.keys():
            foundin = []
            for entry in self._allObjects:
                for daughter in self.children(entry):
                    if self.label(object) == self.label(daughter) and len(self.children(entry)) > 0 and not self.label(entry) in foundin:
                        foundin += [self.label(entry)]
	    try:
                self._foundInDict[object]=foundin
	    except TypeError:
	        return []
        return self._foundInDict[object]

    def usedBy(self, object):
        """ Find config objects that use this as input """
        if not object in self._usedByDict.keys():
            usedby = []
            for entry in self._allObjects:
                for uses in self.uses(entry):
                    if self.label(object) == uses and not self.label(entry) in usedby:
                        usedby += [self.label(entry)]
	    try:
                self._usedByDict[object]=usedby
	    except TypeError:
	        return []
        return self._usedByDict[object]

    def recursePSetProperties(self, name, object, readonly=None):
        #logging.debug(__name__ + ": recursePSetProperties: " + name)
        properties = []
        if name != "" and not isinstance(object, typ.PSet):
            try:
                partyp=str(type(object)).split("'")[1].replace("FWCore.ParameterSet.Types","cms")
                if isinstance(object, cms.InputTag):
                    inputtagValue=object.pythonValue()
                    for i in range(3-len(inputtagValue.split(","))):
                        inputtagValue+=', ""'
                    properties += [("String", name, "cms.InputTag("+inputtagValue+")", partyp, readonly)]
                elif isinstance(object, cms.bool):
                    properties += [("Boolean", name, object.value(), partyp, readonly)]
                elif isinstance(object, (cms.int32, cms.uint32, cms.int64, cms.uint64)):
                    properties += [("Integer", name, object.value(), partyp, readonly)]
                elif isinstance(object, cms.double):
                    properties += [("Double", name, object.value(), partyp, readonly)]
                elif hasattr(object, "pythonValue"):
                    properties += [("String", name, str(object.pythonValue()).strip("\"'"), partyp, readonly)]
                elif hasattr(object, "value"):
                    properties += [("MultilineString", name, str(object.value()), partyp, readonly)]
                else:
                    properties += [("MultilineString", name, str(object), partyp, readonly)]
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
            properties += [("String", "in sequence", text, "This module/sequence is used the listed sequences", True)]
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
                properties += [("MultilineString", "uses", text, "This module/sequence depends on the output of the listes modules/seuquences", True)]
            if len(usedBy) > 0:
                text = ""
                usedby = []
                for entry in usedBy:
                    if text != "":
                        text += ", "
                    text += entry
                properties += [("MultilineString", "used by", text, "The listed modules/sequences depend on the output of this module/sequence", True)]
        if len(self.parameters(object)) > 0:
            properties += [("Category", "Parameters", "")]
            properties += self.recursePSetProperties("", object)
        return tuple(properties)
    
    def setProperty(self, object, name, value, categoryName):
        """ Sets a property with given name to value.
        """
        if hasattr(object, "_seq") and name=="sequence":
            return "Modification of sequences not supported yet."
        else: 
            process=self.process()
            try:
                if isinstance(value,str) and\
                    not value[0]=="[" and\
                    not value[0:4]=="cms.":
                    exec("object." + name + "='''" + value + "'''")
                else:
                    exec("object." + name + "=" + str(value))
            except Exception as e:
                error="Cannot set parameter "+name+" (see logfile for details):\n"+str(e)
                logging.warning(__name__ + ": setProperty: Cannot set parameter "+name+": "+exception_traceback())
                return error
        return True

    def inputEventContent(self):
        content = []
        allLabels = [self.label(object) for object in self._scheduledObjects]
        content_objects = {}
        for object in self._scheduledObjects:
            for key, value in self.inputTags(object):
                elements=str(value).split(":")
                module = elements[0]
                if len(elements)>1:
                    product = elements[1]
                else:
                    product = ""
                if len(elements)>2:
                    process = elements[2]
                else:
                    process = "*"
                if not module in allLabels:
                    if not ("*",module,product,process) in content:
                        content += [("*",module,product,process)]
                    if "*_"+module+"_"+product+"_"+process in content_objects.keys():
                        content_objects["*_"+module+"_"+product+"_"+process]+=","+self.label(object)
                    else:
                        content_objects["*_"+module+"_"+product+"_"+process]=self.label(object)
        return (content,content_objects)

    def outputEventContent(self):
        content = [("*",self.label(object),"*",self.process().process) for object in self._allObjects\
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
        if len(outputModules) > 0 and hasattr(outputModules[0], "outputCommands"):
            return outputModules[0].outputCommands
        else:
            return []

    def applyCommands(self, content, outputCommands):
        keep = {}
        if len(outputCommands)>0 and outputCommands[0]!="keep *":
            for object in content:
                keep[object] = False
        else:
            for object in content:
                keep[object] = True
        for o in outputCommands:
            #skip multiple spaces
            command, filter = [ x for x in o.split(" ") if x]
            if len(filter.split("_")) > 1:
                module = filter.split("_")[1]
                product = filter.split("_")[2]
                process = filter.split("_")[3]
            else:
                module = filter
                product = "*"
                process = "*"
            for object in content:
                if "*" in module:
                    match = module.strip("*") in object[1]
                else:
                    match = module == object[1]
                if "*" in product:
                    match = match and product.strip("*") in object[2]
                else:
                    match = match and product == object[2]
                if "*" in process:
                    match = match and process.strip("*") in object[3]
                else:
                    match = match and process == object[3]
                if match:
                    keep[object] = command == "keep"
        return [object for object in content if keep[object]]
