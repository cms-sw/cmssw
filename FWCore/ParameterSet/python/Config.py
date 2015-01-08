#!/usr/bin/env python

### command line options helper
from  Options import Options
options = Options()


## imports
import sys
from Mixins import PrintOptions,_ParameterTypeBase,_SimpleParameterTypeBase, _Parameterizable, _ConfigureComponent, _TypedParameterizable, _Labelable,  _Unlabelable,  _ValidatingListBase
from Mixins import *
from Types import *
from Modules import *
from Modules import _Module
from SequenceTypes import *
from SequenceTypes import _ModuleSequenceType, _Sequenceable  #extend needs it
from SequenceVisitors import PathValidator, EndPathValidator
from Utilities import *
import DictTypes

from ExceptionHandling import *

#when building RECO paths we have hit the default recursion limit
if sys.getrecursionlimit()<5000:
   sys.setrecursionlimit(5000)

def checkImportPermission(minLevel = 2, allowedPatterns = []):
    """
    Raise an exception if called by special config files. This checks
    the call or import stack for the importing file. An exception is raised if
    the importing module is not in allowedPatterns and if it is called too deeply:
    minLevel = 2: inclusion by top lvel cfg only
    minLevel = 1: No inclusion allowed
    allowedPatterns = ['Module1','Module2/SubModule1'] allows import
    by any module in Module1 or Submodule1
    """

    import inspect
    import os

    ignorePatterns = ['FWCore/ParameterSet/Config.py','<string>']
    CMSSWPath = [os.environ['CMSSW_BASE'],os.environ['CMSSW_RELEASE_BASE']]

    # Filter the stack to things in CMSSWPath and not in ignorePatterns
    trueStack = []
    for item in inspect.stack():
        inPath = False
        ignore = False

        for pattern in CMSSWPath:
            if item[1].find(pattern) != -1:
                inPath = True
                break
        if item[1].find('/') == -1: # The base file, no path
            inPath = True

        for pattern in ignorePatterns:
            if item[1].find(pattern) != -1:
                ignore = True
                break

        if inPath and not ignore:
            trueStack.append(item[1])

    importedFile = trueStack[0]
    importedBy   = ''
    if len(trueStack) > 1:
        importedBy = trueStack[1]

    for pattern in allowedPatterns:
        if importedBy.find(pattern) > -1:
            return True

    if len(trueStack) <= minLevel: # Imported directly
        return True

    raise ImportError("Inclusion of %s is allowed only by cfg or specified cfi files."
                      % importedFile)

def findProcess(module):
    """Look inside the module and find the Processes it contains"""
    class Temp(object):
        pass
    process = None
    if isinstance(module,dict):
        if 'process' in module:
            p = module['process']
            module = Temp()
            module.process = p
    if hasattr(module,'process'):
        if isinstance(module.process,Process):
            process = module.process
        else:
            raise RuntimeError("The attribute named 'process' does not inherit from the Process class")
    else:
        raise RuntimeError("no 'process' attribute found in the module, please add one")
    return process


class Process(object):
    """Root class for a CMS configuration process"""
    def __init__(self,name,*Mods):
        """The argument 'name' will be the name applied to this Process
            Can optionally pass as additional arguments cms.Modifier instances
            which will be used ot modify the Process as it is built
            """
        self.__dict__['_Process__name'] = name
        if not name.isalnum():
            raise RuntimeError("Error: The process name is an empty string or contains non-alphanumeric characters")
        self.__dict__['_Process__filters'] = {}
        self.__dict__['_Process__producers'] = {}
        self.__dict__['_Process__source'] = None
        self.__dict__['_Process__looper'] = None
        self.__dict__['_Process__subProcess'] = None
        self.__dict__['_Process__schedule'] = None
        self.__dict__['_Process__analyzers'] = {}
        self.__dict__['_Process__outputmodules'] = {}
        self.__dict__['_Process__paths'] = DictTypes.SortedKeysDict()    # have to keep the order
        self.__dict__['_Process__endpaths'] = DictTypes.SortedKeysDict() # of definition
        self.__dict__['_Process__sequences'] = {}
        self.__dict__['_Process__services'] = {}
        self.__dict__['_Process__essources'] = {}
        self.__dict__['_Process__esproducers'] = {}
        self.__dict__['_Process__esprefers'] = {}
        self.__dict__['_Process__aliases'] = {}
        self.__dict__['_Process__psets']={}
        self.__dict__['_Process__vpsets']={}
        self.__dict__['_cloneToObjectDict'] = {}
        # policy switch to avoid object overwriting during extend/load
        self.__dict__['_Process__InExtendCall'] = False
        self.__dict__['_Process__partialschedules'] = {}
        self.__isStrict = False
        self.__dict__['_Process__modifiers'] = Mods
        for m in self.__modifiers:
            m._setChosen()

    def setStrict(self, value):
        self.__isStrict = value
        _Module.__isStrict__ = True

    # some user-friendly methods for command-line browsing
    def producerNames(self):
        """Returns a string containing all the EDProducer labels separated by a blank"""
        return ' '.join(self.producers_().keys())
    def analyzerNames(self):
        """Returns a string containing all the EDAnalyzer labels separated by a blank"""
        return ' '.join(self.analyzers_().keys())
    def filterNames(self):
        """Returns a string containing all the EDFilter labels separated by a blank"""
        return ' '.join(self.filters_().keys())
    def pathNames(self):
        """Returns a string containing all the Path names separated by a blank"""
        return ' '.join(self.paths_().keys())

    def __setstate__(self, pkldict):
        """
        Unpickling hook.

        Since cloneToObjectDict stores a hash of objects by their
        id() it needs to be updated when unpickling to use the
        new object id values instantiated during the unpickle.

        """
        self.__dict__.update(pkldict)
        tmpDict = {}
        for value in self._cloneToObjectDict.values():
            tmpDict[id(value)] = value
        self.__dict__['_cloneToObjectDict'] = tmpDict



    def filters_(self):
        """returns a dict of the filters which have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__filters)
    filters = property(filters_, doc="dictionary containing the filters for the process")
    def name_(self):
        return self.__name
    def setName_(self,name):
        if not name.isalnum():
            raise RuntimeError("Error: The process name is an empty string or contains non-alphanumeric characters")
        self.__dict__['_Process__name'] = name
    process = property(name_,setName_, doc="name of the process")
    def producers_(self):
        """returns a dict of the producers which have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__producers)
    producers = property(producers_,doc="dictionary containing the producers for the process")
    def source_(self):
        """returns the source which has been added to the Process or None if none have been added"""
        return self.__source
    def setSource_(self,src):
        self._placeSource('source',src)
    source = property(source_,setSource_,doc='the main source or None if not set')
    def looper_(self):
        """returns the looper which has been added to the Process or None if none have been added"""
        return self.__looper
    def setLooper_(self,lpr):
        self._placeLooper('looper',lpr)
    looper = property(looper_,setLooper_,doc='the main looper or None if not set')
    def subProcess_(self):
        """returns the sub-process which has been added to the Process or None if none have been added"""
        return self.__subProcess
    def setSubProcess_(self,lpr):
        self._placeSubProcess('subProcess',lpr)
    subProcess = property(subProcess_,setSubProcess_,doc='the SubProcess or None if not set')
    def analyzers_(self):
        """returns a dict of the analyzers which have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__analyzers)
    analyzers = property(analyzers_,doc="dictionary containing the analyzers for the process")
    def outputModules_(self):
        """returns a dict of the output modules which have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__outputmodules)
    outputModules = property(outputModules_,doc="dictionary containing the output_modules for the process")
    def paths_(self):
        """returns a dict of the paths which have been added to the Process"""
        return DictTypes.SortedAndFixedKeysDict(self.__paths)
    paths = property(paths_,doc="dictionary containing the paths for the process")
    def endpaths_(self):
        """returns a dict of the endpaths which have been added to the Process"""
        return DictTypes.SortedAndFixedKeysDict(self.__endpaths)
    endpaths = property(endpaths_,doc="dictionary containing the endpaths for the process")
    def sequences_(self):
        """returns a dict of the sequences which have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__sequences)
    sequences = property(sequences_,doc="dictionary containing the sequences for the process")
    def schedule_(self):
        """returns the schedule which has been added to the Process or None if none have been added"""
        return self.__schedule
    def setPartialSchedule_(self,sch,label):
        if label == "schedule":
            self.setSchedule_(sch)
        else:
            self._place(label, sch, self.__partialschedules)
    def setSchedule_(self,sch):
                # See if every module has been inserted into the process
        index = 0
        try:
            for p in sch:
               p.label_()
               index +=1
        except:
            raise RuntimeError("The path at index "+str(index)+" in the Schedule was not attached to the process.")

        self.__dict__['_Process__schedule'] = sch
    schedule = property(schedule_,setSchedule_,doc='the schedule or None if not set')
    def services_(self):
        """returns a dict of the services which have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__services)
    services = property(services_,doc="dictionary containing the services for the process")
    def es_producers_(self):
        """returns a dict of the esproducers which have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__esproducers)
    es_producers = property(es_producers_,doc="dictionary containing the es_producers for the process")
    def es_sources_(self):
        """returns a the es_sources which have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__essources)
    es_sources = property(es_sources_,doc="dictionary containing the es_sources for the process")
    def es_prefers_(self):
        """returns a dict of the es_prefers which have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__esprefers)
    es_prefers = property(es_prefers_,doc="dictionary containing the es_prefers for the process")
    def aliases_(self):
        """returns a dict of the aliases that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__aliases)
    aliases = property(aliases_,doc="dictionary containing the aliases for the process")
    def psets_(self):
        """returns a dict of the PSets which have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__psets)
    psets = property(psets_,doc="dictionary containing the PSets for the process")
    def vpsets_(self):
        """returns a dict of the VPSets which have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__vpsets)
    vpsets = property(vpsets_,doc="dictionary containing the PSets for the process")

    def __setObjectLabel(self, object, newLabel) :
        if not object.hasLabel_() :
            object.setLabel(newLabel)
            return
        if newLabel == object.label_() :
            return
        if newLabel is None :
            object.setLabel(None)
            return
        if (hasattr(self, object.label_()) and id(getattr(self, object.label_())) == id(object)) :
            msg100 = "Attempting to change the label of an attribute of the Process\n"
            msg101 = "Old label = "+object.label_()+"  New label = "+newLabel+"\n"
            msg102 = "Type = "+str(type(object))+"\n"
            msg103 = "Some possible solutions:\n"
            msg104 = "  1. Clone modules instead of using simple assignment. Cloning is\n"
            msg105 = "  also preferred for other types when possible.\n"
            msg106 = "  2. Declare new names starting with an underscore if they are\n"
            msg107 = "  for temporaries you do not want propagated into the Process. The\n"
            msg108 = "  underscore tells \"from x import *\" and process.load not to import\n"
            msg109 = "  the name.\n"
            msg110 = "  3. Reorganize so the assigment is not necessary. Giving a second\n"
            msg111 = "  name to the same object usually causes confusion and problems.\n"
            msg112 = "  4. Compose Sequences: newName = cms.Sequence(oldName)\n"
            raise ValueError(msg100+msg101+msg102+msg103+msg104+msg105+msg106+msg107+msg108+msg109+msg110+msg111+msg112)
        object.setLabel(None)
        object.setLabel(newLabel)

    def __setattr__(self,name,value):
        # check if the name is well-formed (only _ and alphanumerics are allowed)
        if not name.replace('_','').isalnum():
            raise ValueError('The label '+name+' contains forbiden characters')

        # private variable exempt from all this
        if name.startswith('_Process__'):
            self.__dict__[name]=value
            return
        if not isinstance(value,_ConfigureComponent):
            raise TypeError("can only assign labels to an object which inherits from '_ConfigureComponent'\n"
                            +"an instance of "+str(type(value))+" will not work")
        if not isinstance(value,_Labelable) and not isinstance(value,Source) and not isinstance(value,Looper) and not isinstance(value,Schedule):
            if name == value.type_():
                self.add_(value)
                return
            else:
                raise TypeError("an instance of "+str(type(value))+" can not be assigned the label '"+name+"'.\n"+
                                "Please either use the label '"+value.type_()+" or use the 'add_' method instead.")
        #clone the item
        if self.__isStrict:
            newValue =value.copy()
            try:
              newValue._filename = value._filename
            except:
              pass
            value.setIsFrozen()
        else:
            newValue =value
        if not self._okToPlace(name, value, self.__dict__):
            newFile='top level config'
            if hasattr(value,'_filename'):
               newFile = value._filename
            oldFile='top level config'
            oldValue = getattr(self,name)
            if hasattr(oldValue,'_filename'):
               oldFile = oldValue._filename
            msg = "Trying to override definition of process."+name
            msg += "\n new object defined in: "+newFile
            msg += "\n existing object defined in: "+oldFile
            raise ValueError(msg)
        # remove the old object of the name (if there is one)
        if hasattr(self,name) and not (getattr(self,name)==newValue):
            # Compain if items in sequences from load() statements have
            # degeneratate names, but if the user overwrites a name in the
            # main config, replace it everywhere
            if isinstance(newValue, _Sequenceable):
                if not self.__InExtendCall:
                   self._replaceInSequences(name, newValue)
                else:
                   #should check to see if used in sequence before complaining
                   newFile='top level config'
                   if hasattr(value,'_filename'):
                      newFile = value._filename
                   oldFile='top level config'
                   oldValue = getattr(self,name)
                   if hasattr(oldValue,'_filename'):
                      oldFile = oldValue._filename
                   msg1 = "Trying to override definition of "+name+" while it is used by the sequence "
                   msg2 = "\n new object defined in: "+newFile
                   msg2 += "\n existing object defined in: "+oldFile
                   s = self.__findFirstSequenceUsingModule(self.sequences,oldValue)
                   if s is not None:
                      raise ValueError(msg1+s.label_()+msg2)
                   s = self.__findFirstSequenceUsingModule(self.paths,oldValue)
                   if s is not None:
                      raise ValueError(msg1+s.label_()+msg2)
                   s = self.__findFirstSequenceUsingModule(self.endpaths,oldValue)
                   if s is not None:
                      raise ValueError(msg1+s.label_()+msg2)
            self.__delattr__(name)
        self.__dict__[name]=newValue
        if isinstance(newValue,_Labelable):
            self.__setObjectLabel(newValue, name)
            self._cloneToObjectDict[id(value)] = newValue
            self._cloneToObjectDict[id(newValue)] = newValue
        #now put in proper bucket
        newValue._place(name,self)
    def __findFirstSequenceUsingModule(self,seqs,mod):
       """Given a container of sequences, find the first sequence containing mod
       and return the sequence. If no sequence is found, return None"""
       from FWCore.ParameterSet.SequenceTypes import ModuleNodeVisitor
       for sequenceable in seqs.itervalues():
          l = list()
          v = ModuleNodeVisitor(l)
          sequenceable.visit(v)
          if mod in l:
             return sequenceable
       return None
    def __delattr__(self,name):
        if not hasattr(self,name):
            raise KeyError('process does not know about '+name)
        elif name.startswith('_Process__'):
            raise ValueError('this attribute cannot be deleted')
        else:
            # we have to remove it from all dictionaries/registries
            dicts = [item for item in self.__dict__.values() if (type(item)==dict or type(item)==DictTypes.SortedKeysDict)]
            for reg in dicts:
                if reg.has_key(name): del reg[name]
            # if it was a labelable object, the label needs to be removed
            obj = getattr(self,name)
            if isinstance(obj,_Labelable):
                getattr(self,name).setLabel(None)
            # now remove it from the process itself
            try:
                del self.__dict__[name]
            except:
                pass

    def add_(self,value):
        """Allows addition of components which do not have to have a label, e.g. Services"""
        if not isinstance(value,_ConfigureComponent):
            raise TypeError
        if not isinstance(value,_Unlabelable):
            raise TypeError
        #clone the item
        #clone the item
        if self.__isStrict:
            newValue =value.copy()
            value.setIsFrozen()
        else:
            newValue =value
        newValue._place('',self)

    def _okToPlace(self, name, mod, d):
        if not self.__InExtendCall:
            # if going
            return True
        elif not self.__isStrict:
            return True
        elif name in d:
            # if there's an old copy, and the new one
            # hasn't been modified, we're done.  Still
            # not quite safe if something has been defined twice.
            #  Need to add checks
            if mod._isModified:
                if d[name]._isModified:
                    return False
                else:
                    return True
            else:
                return True
        else:
            return True

    def _place(self, name, mod, d):
        if self._okToPlace(name, mod, d):
            if self.__isStrict and isinstance(mod, _ModuleSequenceType):
                d[name] = mod._postProcessFixup(self._cloneToObjectDict)
            else:
                d[name] = mod
            if isinstance(mod,_Labelable):
                self.__setObjectLabel(mod, name)
    def _placeOutputModule(self,name,mod):
        self._place(name, mod, self.__outputmodules)
    def _placeProducer(self,name,mod):
        self._place(name, mod, self.__producers)
    def _placeFilter(self,name,mod):
        self._place(name, mod, self.__filters)
    def _placeAnalyzer(self,name,mod):
        self._place(name, mod, self.__analyzers)
    def _placePath(self,name,mod):
        self._validateSequence(mod, name)
        try:
            self._place(name, mod, self.__paths)
        except ModuleCloneError, msg:
            context = format_outerframe(4)
            raise Exception("%sThe module %s in path %s is unknown to the process %s." %(context, msg, name, self._Process__name))
    def _placeEndPath(self,name,mod):
        self._validateSequence(mod, name)
        try:
            self._place(name, mod, self.__endpaths)
        except ModuleCloneError, msg:
            context = format_outerframe(4)
            raise Exception("%sThe module %s in endpath %s is unknown to the process %s." %(context, msg, name, self._Process__name))
    def _placeSequence(self,name,mod):
        self._validateSequence(mod, name)
        self._place(name, mod, self.__sequences)
    def _placeESProducer(self,name,mod):
        self._place(name, mod, self.__esproducers)
    def _placeESPrefer(self,name,mod):
        self._place(name, mod, self.__esprefers)
    def _placeESSource(self,name,mod):
        self._place(name, mod, self.__essources)
    def _placeAlias(self,name,mod):
        self._place(name, mod, self.__aliases)
    def _placePSet(self,name,mod):
        self._place(name, mod, self.__psets)
    def _placeVPSet(self,name,mod):
        self._place(name, mod, self.__vpsets)
    def _placeSource(self,name,mod):
        """Allow the source to be referenced by 'source' or by type name"""
        if name != 'source':
            raise ValueError("The label '"+name+"' can not be used for a Source.  Only 'source' is allowed.")
        if self.__dict__['_Process__source'] is not None :
            del self.__dict__[self.__dict__['_Process__source'].type_()]
        self.__dict__['_Process__source'] = mod
        self.__dict__[mod.type_()] = mod
    def _placeLooper(self,name,mod):
        if name != 'looper':
            raise ValueError("The label '"+name+"' can not be used for a Looper.  Only 'looper' is allowed.")
        self.__dict__['_Process__looper'] = mod
        self.__dict__[mod.type_()] = mod
    def _placeSubProcess(self,name,mod):
        if name != 'subProcess':
            raise ValueError("The label '"+name+"' can not be used for a SubProcess.  Only 'subProcess' is allowed.")
        self.__dict__['_Process__subProcess'] = mod
        self.__dict__[mod.type_()] = mod
    def _placeService(self,typeName,mod):
        self._place(typeName, mod, self.__services)
        self.__dict__[typeName]=mod
    def load(self, moduleName):
        moduleName = moduleName.replace("/",".")
        module = __import__(moduleName)
        self.extend(sys.modules[moduleName])
    def extend(self,other,items=()):
        """Look in other and find types which we can use"""
        # enable explicit check to avoid overwriting of existing objects
        self.__dict__['_Process__InExtendCall'] = True

        seqs = dict()
        for name in dir(other):
            #'from XX import *' ignores these, and so should we.
            if name.startswith('_'):
                continue
            item = getattr(other,name)
            if name == "source" or name == "looper" or name == "subProcess":
                self.__setattr__(name,item)
            elif isinstance(item,_ModuleSequenceType):
                seqs[name]=item
            elif isinstance(item,_Labelable):
                self.__setattr__(name,item)
                if not item.hasLabel_() :
                    item.setLabel(name)
            elif isinstance(item,Schedule):
                self.__setattr__(name,item)
            elif isinstance(item,_Unlabelable):
                self.add_(item)
            elif isinstance(item,ProcessModifier):
                item.apply(self)

        #now create a sequence which uses the newly made items
        for name in seqs.iterkeys():
            seq = seqs[name]
            #newSeq = seq.copy()
            #
            if id(seq) not in self._cloneToObjectDict:
                self.__setattr__(name,seq)
            else:
                newSeq = self._cloneToObjectDict[id(seq)]
                self.__dict__[name]=newSeq
                self.__setObjectLabel(newSeq, name)
                #now put in proper bucket
                newSeq._place(name,self)
        self.__dict__['_Process__InExtendCall'] = False

    def _dumpConfigNamedList(self,items,typeName,options):
        returnValue = ''
        for name,item in items:
            returnValue +=options.indentation()+typeName+' '+name+' = '+item.dumpConfig(options)
        return returnValue
    def _dumpConfigUnnamedList(self,items,typeName,options):
        returnValue = ''
        for name,item in items:
            returnValue +=options.indentation()+typeName+' = '+item.dumpConfig(options)
        return returnValue
    def _dumpConfigOptionallyNamedList(self,items,typeName,options):
        returnValue = ''
        for name,item in items:
            if name == item.type_():
                name = ''
            returnValue +=options.indentation()+typeName+' '+name+' = '+item.dumpConfig(options)
        return returnValue
    def dumpConfig(self, options=PrintOptions()):
        """return a string containing the equivalent process defined using the old configuration language"""
        config = "process "+self.__name+" = {\n"
        options.indent()
        if self.source_():
            config += options.indentation()+"source = "+self.source_().dumpConfig(options)
        if self.looper_():
            config += options.indentation()+"looper = "+self.looper_().dumpConfig(options)
        if self.subProcess_():
            config += options.indentation()+"subProcess = "+self.subProcess_().dumpConfig(options)

        config+=self._dumpConfigNamedList(self.producers_().iteritems(),
                                  'module',
                                  options)
        config+=self._dumpConfigNamedList(self.filters_().iteritems(),
                                  'module',
                                  options)
        config+=self._dumpConfigNamedList(self.analyzers_().iteritems(),
                                  'module',
                                  options)
        config+=self._dumpConfigNamedList(self.outputModules_().iteritems(),
                                  'module',
                                  options)
        config+=self._dumpConfigNamedList(self.sequences_().iteritems(),
                                  'sequence',
                                  options)
        config+=self._dumpConfigNamedList(self.paths_().iteritems(),
                                  'path',
                                  options)
        config+=self._dumpConfigNamedList(self.endpaths_().iteritems(),
                                  'endpath',
                                  options)
        config+=self._dumpConfigUnnamedList(self.services_().iteritems(),
                                  'service',
                                  options)
        config+=self._dumpConfigNamedList(self.aliases_().iteritems(),
                                  'alias',
                                  options)
        config+=self._dumpConfigOptionallyNamedList(
            self.es_producers_().iteritems(),
            'es_module',
            options)
        config+=self._dumpConfigOptionallyNamedList(
            self.es_sources_().iteritems(),
            'es_source',
            options)
        config += self._dumpConfigESPrefers(options)
        for name,item in self.psets.iteritems():
            config +=options.indentation()+item.configTypeName()+' '+name+' = '+item.configValue(options)
        for name,item in self.vpsets.iteritems():
            config +=options.indentation()+'VPSet '+name+' = '+item.configValue(options)
        if self.schedule:
            pathNames = [p.label_() for p in self.schedule]
            config +=options.indentation()+'schedule = {'+','.join(pathNames)+'}\n'

#        config+=self._dumpConfigNamedList(self.vpsets.iteritems(),
#                                  'VPSet',
#                                  options)
        config += "}\n"
        options.unindent()
        return config
    def _dumpConfigESPrefers(self, options):
        result = ''
        for item in self.es_prefers_().itervalues():
            result +=options.indentation()+'es_prefer '+item.targetLabel_()+' = '+item.dumpConfig(options)
        return result
    def _dumpPythonList(self, d, options):
        returnValue = ''
        if isinstance(d, DictTypes.SortedKeysDict):
            for name,item in d.items():
                returnValue +='process.'+name+' = '+item.dumpPython(options)+'\n\n'
        else:
            for name,item in sorted(d.items()):
                returnValue +='process.'+name+' = '+item.dumpPython(options)+'\n\n'
        return returnValue
    def _validateSequence(self, sequence, label):
        # See if every module has been inserted into the process
        try:
            l = set()
            nameVisitor = NodeNameVisitor(l)
            sequence.visit(nameVisitor)
        except:
            raise RuntimeError("An entry in sequence "+label + ' has no label')
    def _sequencesInDependencyOrder(self):
        #for each sequence, see what other sequences it depends upon
        returnValue=DictTypes.SortedKeysDict()
        dependencies = {}
        for label,seq in self.sequences.iteritems():
            d = []
            v = SequenceVisitor(d)
            seq.visit(v)
            dependencies[label]=[dep.label_() for dep in d if dep.hasLabel_()]
        resolvedDependencies=True
        #keep looping until we can no longer get rid of all dependencies
        # if that happens it means we have circular dependencies
        iterCount = 0
        while resolvedDependencies:
            iterCount += 1
            resolvedDependencies = (0 != len(dependencies))
            oldDeps = dict(dependencies)
            for label,deps in oldDeps.iteritems():
                # don't try too hard
                if len(deps)==0 or iterCount > 100:
                    iterCount = 0
                    resolvedDependencies=True
                    returnValue[label]=self.sequences[label]
                    #remove this as a dependency for all other sequences
                    del dependencies[label]
                    for lb2,deps2 in dependencies.iteritems():
                        while deps2.count(label):
                            deps2.remove(label)
        if len(dependencies):
            raise RuntimeError("circular sequence dependency discovered \n"+
                               ",".join([label for label,junk in dependencies.iteritems()]))
        return returnValue
    def _dumpPython(self, d, options):
        result = ''
        for name, value in sorted(d.iteritems()):
           result += value.dumpPythonAs(name,options)+'\n'
        return result
    def dumpPython(self, options=PrintOptions()):
        """return a string containing the equivalent process defined using python"""
        result = "import FWCore.ParameterSet.Config as cms\n\n"
        result += "process = cms.Process(\""+self.__name+"\")\n\n"
        if self.source_():
            result += "process.source = "+self.source_().dumpPython(options)
        if self.looper_():
            result += "process.looper = "+self.looper_().dumpPython()
        if self.subProcess_():
            result += self.subProcess_().dumpPython(options)
        result+=self._dumpPythonList(self.producers_(), options)
        result+=self._dumpPythonList(self.filters_() , options)
        result+=self._dumpPythonList(self.analyzers_(), options)
        result+=self._dumpPythonList(self.outputModules_(), options)
        result+=self._dumpPythonList(self._sequencesInDependencyOrder(), options)
        result+=self._dumpPythonList(self.paths_(), options)
        result+=self._dumpPythonList(self.endpaths_(), options)
        result+=self._dumpPythonList(self.services_(), options)
        result+=self._dumpPythonList(self.es_producers_(), options)
        result+=self._dumpPythonList(self.es_sources_(), options)
        result+=self._dumpPython(self.es_prefers_(), options)
        result+=self._dumpPythonList(self.aliases_(), options)
        result+=self._dumpPythonList(self.psets, options)
        result+=self._dumpPythonList(self.vpsets, options)
        if self.schedule:
            pathNames = ['process.'+p.label_() for p in self.schedule]
            result +='process.schedule = cms.Schedule(*[ ' + ', '.join(pathNames) + ' ])\n'

        return result
    def _replaceInSequences(self, label, new):
        old = getattr(self,label)
        #TODO - replace by iterator concatenation
        for sequenceable in self.sequences.itervalues():
            sequenceable.replace(old,new)
        for sequenceable in self.paths.itervalues():
            sequenceable.replace(old,new)
        for sequenceable in self.endpaths.itervalues():
            sequenceable.replace(old,new)
    def globalReplace(self,label,new):
        """ Replace the item with label 'label' by object 'new' in the process and all sequences/paths"""
        if not hasattr(self,label):
            raise LookupError("process has no item of label "+label)
        self._replaceInSequences(label, new)
        setattr(self,label,new)
    def _insertInto(self, parameterSet, itemDict):
        for name,value in itemDict.iteritems():
            value.insertInto(parameterSet, name)
    def _insertOneInto(self, parameterSet, label, item, tracked):
        vitems = []
        if not item == None:
            newlabel = item.nameInProcessDesc_(label)
            vitems = [newlabel]
            item.insertInto(parameterSet, newlabel)
        parameterSet.addVString(tracked, label, vitems)
    def _insertManyInto(self, parameterSet, label, itemDict, tracked):
        l = []
        for name,value in itemDict.iteritems():
          newLabel = value.nameInProcessDesc_(name)
          l.append(newLabel)
          value.insertInto(parameterSet, name)
        # alphabetical order is easier to compare with old language
        l.sort()
        parameterSet.addVString(tracked, label, l)
    def _insertPaths(self, processPSet):
        scheduledPaths = []
        triggerPaths = []
        endpaths = []
        if self.schedule_() == None:
            # make one from triggerpaths & endpaths
            for name,value in self.paths_().iteritems():
                scheduledPaths.append(name)
                triggerPaths.append(name)
            for name,value in self.endpaths_().iteritems():
                scheduledPaths.append(name)
                endpaths.append(name)
        else:
            for path in self.schedule_():
               pathname = path.label_()
               scheduledPaths.append(pathname)
               if self.endpaths_().has_key(pathname):
                   endpaths.append(pathname)
               else:
                   triggerPaths.append(pathname)
        processPSet.addVString(True, "@end_paths", endpaths)
        processPSet.addVString(True, "@paths", scheduledPaths)
        # trigger_paths are a little different
        p = processPSet.newPSet()
        p.addVString(True, "@trigger_paths", triggerPaths)
        processPSet.addPSet(True, "@trigger_paths", p)
        # add all these paths
        pathValidator = PathValidator()
        endpathValidator = EndPathValidator()
        for triggername in triggerPaths:
            #self.paths_()[triggername].insertInto(processPSet, triggername, self.sequences_())
            pathValidator.setLabel(triggername)
            self.paths_()[triggername].visit(pathValidator)
            self.paths_()[triggername].insertInto(processPSet, triggername, self.__dict__)
        for endpathname in endpaths:
            #self.endpaths_()[endpathname].insertInto(processPSet, endpathname, self.sequences_())
            endpathValidator.setLabel(endpathname)
            self.endpaths_()[endpathname].visit(endpathValidator)
            self.endpaths_()[endpathname].insertInto(processPSet, endpathname, self.__dict__)
        processPSet.addVString(False, "@filters_on_endpaths", endpathValidator.filtersOnEndpaths)

    def prune(self,verbose=False):
        """ Remove clutter from the process which we think is unnecessary:
        tracked PSets, VPSets and unused modules and sequences. If a Schedule has been set, then Paths and EndPaths
        not in the schedule will also be removed, along with an modules and sequences used only by
        those removed Paths and EndPaths."""
# need to update this to only prune psets not on refToPSets
# but for now, remove the delattr
#        for name in self.psets_():
#            if getattr(self,name).isTracked():
#                delattr(self, name)
        for name in self.vpsets_():
            delattr(self, name)
        #first we need to resolve any SequencePlaceholders being used
        for x in self.paths.itervalues():
            x.resolve(self.__dict__)
        for x in self.endpaths.itervalues():
            x.resolve(self.__dict__)
        usedModules = set()
        unneededPaths = set()
        if self.schedule_():
            usedModules=set(self.schedule_().moduleNames())
            #get rid of unused paths
            schedNames = set(( x.label_() for x in self.schedule_()))
            names = set(self.paths)
            names.update(set(self.endpaths))
            unneededPaths = names - schedNames
            for n in unneededPaths:
                delattr(self,n)
        else:
            pths = list(self.paths.itervalues())
            pths.extend(self.endpaths.itervalues())
            temp = Schedule(*pths)
            usedModules=set(temp.moduleNames())
        unneededModules = self._pruneModules(self.producers_(), usedModules)
        unneededModules.update(self._pruneModules(self.filters_(), usedModules))
        unneededModules.update(self._pruneModules(self.analyzers_(), usedModules))
        #remove sequences that do not appear in remaining paths and endpaths
        seqs = list()
        sv = SequenceVisitor(seqs)
        for p in self.paths.itervalues():
            p.visit(sv)
        for p in self.endpaths.itervalues():
            p.visit(sv)
        keepSeqSet = set(( s for s in seqs if s.hasLabel_()))
        availableSeqs = set(self.sequences.itervalues())
        unneededSeqs = availableSeqs-keepSeqSet
        unneededSeqLabels = []
        for s in unneededSeqs:
            unneededSeqLabels.append(s.label_())
            delattr(self,s.label_())
        if verbose:
            print "prune removed the following:"
            print "  modules:"+",".join(unneededModules)
            print "  sequences:"+",".join(unneededSeqLabels)
            print "  paths/endpaths:"+",".join(unneededPaths)
    def _pruneModules(self, d, scheduledNames):
        moduleNames = set(d.keys())
        junk = moduleNames - scheduledNames
        for name in junk:
            delattr(self, name)
        return junk

    def fillProcessDesc(self, processPSet):
        """Used by the framework to convert python to C++ objects"""
        class ServiceInjectorAdaptor(object):
            def __init__(self,ppset,thelist):
                self.__thelist = thelist
                self.__processPSet = ppset
            def addService(self,pset):
                self.__thelist.append(pset)
            def newPSet(self):
                return self.__processPSet.newPSet()
        #This adaptor is used to 'add' the method 'getTopPSet_'
        # to the ProcessDesc and PythonParameterSet C++ classes.
        # This method is needed for the PSet refToPSet_ functionality.
        class TopLevelPSetAcessorAdaptor(object):
            def __init__(self,ppset,process):
                self.__ppset = ppset
                self.__process = process
            def __getattr__(self,attr):
                return getattr(self.__ppset,attr)
            def getTopPSet_(self,label):
                return getattr(self.__process,label)
            def newPSet(self):
                return TopLevelPSetAcessorAdaptor(self.__ppset.newPSet(),self.__process)
            def addPSet(self,tracked,name,ppset):
                return self.__ppset.addPSet(tracked,name,self.__extractPSet(ppset))
            def addVPSet(self,tracked,name,vpset):
                return self.__ppset.addVPSet(tracked,name,[self.__extractPSet(x) for x in vpset])
            def __extractPSet(self,pset):
                if isinstance(pset,TopLevelPSetAcessorAdaptor):
                    return pset.__ppset
                return pset
                
        self.validate()
        processPSet.addString(True, "@process_name", self.name_())
        all_modules = self.producers_().copy()
        all_modules.update(self.filters_())
        all_modules.update(self.analyzers_())
        all_modules.update(self.outputModules_())
        adaptor = TopLevelPSetAcessorAdaptor(processPSet,self)
        self._insertInto(adaptor, self.psets_())
        self._insertInto(adaptor, self.vpsets_())
        self._insertManyInto(adaptor, "@all_modules", all_modules, True)
        self._insertOneInto(adaptor,  "@all_sources", self.source_(), True)
        self._insertOneInto(adaptor,  "@all_loopers", self.looper_(), True)
        self._insertOneInto(adaptor,  "@all_subprocesses", self.subProcess_(), False)
        self._insertManyInto(adaptor, "@all_esmodules", self.es_producers_(), True)
        self._insertManyInto(adaptor, "@all_essources", self.es_sources_(), True)
        self._insertManyInto(adaptor, "@all_esprefers", self.es_prefers_(), True)
        self._insertManyInto(adaptor, "@all_aliases", self.aliases_(), True)
        self._insertPaths(adaptor)
        #handle services differently
        services = []
        for n in self.services_():
             getattr(self,n).insertInto(ServiceInjectorAdaptor(adaptor,services))
        adaptor.addVPSet(False,"services",services)
        return processPSet

    def validate(self):
        # check if there's some input
        # Breaks too many unit tests for now
        #if self.source_() == None and self.looper_() == None:
        #    raise RuntimeError("No input source was found for this process")
        pass

    def prefer(self, esmodule,*args,**kargs):
        """Prefer this ES source or producer.  The argument can
           either be an object label, e.g.,
             process.prefer(process.juicerProducer) (not supported yet)
           or a name of an ESSource or ESProducer
             process.prefer("juicer")
           or a type of unnamed ESSource or ESProducer
             process.prefer("JuicerProducer")
           In addition, you can pass as a labelled arguments the name of the Record you wish to
           prefer where the type passed is a cms.vstring and that vstring can contain the
           name of the C++ types in the Record which are being preferred, e.g.,
              #prefer all data in record 'OrangeRecord' from 'juicer'
              process.prefer("juicer", OrangeRecord=cms.vstring())
           or
              #prefer only "Orange" data in "OrangeRecord" from "juicer"
              process.prefer("juicer", OrangeRecord=cms.vstring("Orange"))
           or
              #prefer only "Orange" data with label "ExtraPulp" in "OrangeRecord" from "juicer"
              ESPrefer("ESJuicerProd", OrangeRecord=cms.vstring("Orange/ExtraPulp"))
        """
        # see if this refers to a named ESProducer
        if isinstance(esmodule, ESSource) or isinstance(esmodule, ESProducer):
            raise RuntimeError("Syntax of process.prefer(process.esmodule) not supported yet")
        elif self._findPreferred(esmodule, self.es_producers_(),*args,**kargs) or \
                self._findPreferred(esmodule, self.es_sources_(),*args,**kargs):
            pass
        else:
            raise RuntimeError("Cannot resolve prefer for "+repr(esmodule))

    def _findPreferred(self, esname, d,*args,**kargs):
        # is esname a name in the dictionary?
        if esname in d:
            typ = d[esname].type_()
            if typ == esname:
                self.__setattr__( esname+"_prefer", ESPrefer(typ,*args,**kargs) )
            else:
                self.__setattr__( esname+"_prefer", ESPrefer(typ, esname,*args,**kargs) )
            return True
        else:
            # maybe it's an unnamed ESModule?
            found = False
            for name, value in d.iteritems():
               if value.type_() == esname:
                  if found:
                      raise RuntimeError("More than one ES module for "+esname)
                  found = True
                  self.__setattr__(esname+"_prefer",  ESPrefer(d[esname].type_()) )
            return found

class FilteredStream(dict):
    """a dictionary with fixed keys"""
    def _blocked_attribute(obj):
        raise AttributeError, "An FilteredStream defintion cannot be modified after creation."
    _blocked_attribute = property(_blocked_attribute)
    __setattr__ = __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute
    def __new__(cls, *args, **kw):
        new = dict.__new__(cls)
        dict.__init__(new, *args, **kw)
        keys = kw.keys()
        keys.sort()
        if keys != ['content', 'dataTier', 'name', 'paths', 'responsible', 'selectEvents']:
           raise ValueError("The needed parameters are: content, dataTier, name, paths, responsible, selectEvents")
	if not isinstance(kw['name'],str):
           raise ValueError("name must be of type string")
        if not isinstance(kw['content'], vstring) and not isinstance(kw['content'],str):
           raise ValueError("content must be of type vstring or string")
        if not isinstance(kw['dataTier'], string):
           raise ValueError("dataTier must be of type string")
        if not isinstance(kw['selectEvents'], PSet):
           raise ValueError("selectEvents must be of type PSet")
        if not isinstance(kw['paths'],(tuple, Path)):
           raise ValueError("'paths' must be a tuple of paths")
	return new
    def __init__(self, *args, **kw):
        pass
    def __repr__(self):
        return "FilteredStream object: %s" %self["name"]
    def __getattr__(self,attr):
        return self[attr]

class SubProcess(_ConfigureComponent,_Unlabelable):
   """Allows embedding another process within a parent process. This allows one to 
   chain processes together directly in one cmsRun job rather than having to run
   separate jobs which are connected via a temporary file.
   """
   def __init__(self,process, SelectEvents = untracked.PSet(), outputCommands = untracked.vstring()):
      """
      """
      if not isinstance(process, Process):
         raise ValueError("the 'process' argument must be of type cms.Process")
      if not isinstance(SelectEvents,PSet):
         raise ValueError("the 'SelectEvents' argument must be of type cms.untracked.PSet")
      if not isinstance(outputCommands,vstring):
         raise ValueError("the 'outputCommands' argument must be of type cms.untracked.vstring")
      self.__process = process
      self.__SelectEvents = SelectEvents
      self.__outputCommands = outputCommands
   def dumpPython(self,options):
      out = "parentProcess"+str(hash(self))+" = process\n"
      out += self.__process.dumpPython()
      out += "childProcess = process\n"
      out += "process = parentProcess"+str(hash(self))+"\n"
      out += "process.subProcess = cms.SubProcess( process = childProcess, SelectEvents = "+self.__SelectEvents.dumpPython(options) +", outputCommands = "+self.__outputCommands.dumpPython(options) +")\n"
      return out
   def type_(self):
      return 'subProcess'
   def nameInProcessDesc_(self,label):
      return '@sub_process'
   def _place(self,label,process):
      process._placeSubProcess('subProcess',self)
   def insertInto(self,parameterSet, newlabel):
      topPSet = parameterSet.newPSet()
      self.__process.fillProcessDesc(topPSet)
      subProcessPSet = parameterSet.newPSet()
      self.__SelectEvents.insertInto(subProcessPSet,"SelectEvents")
      self.__outputCommands.insertInto(subProcessPSet,"outputCommands")
      subProcessPSet.addPSet(False,"process",topPSet)
      parameterSet.addPSet(False,self.nameInProcessDesc_("subProcess"), subProcessPSet)

class _ParameterModifier(object):
  """Helper class for Modifier which takes key/value pairs and uses them to reset parameters of the object"""
  def __init__(self,args):
    self.__args = args
  def __call__(self,obj):
    for k,v in self.__args.iteritems():
      setattr(obj,k,v)

class Modifier(object):
  """This class is used to define standard modifications to a Process.
  An instance of this class is declared to denote a specific modification,e.g. era2017 could
  reconfigure items in a process to match our expectation of running in 2017. Once declared,
  these Modifier instances are imported into a configuration and items which need to be modified
  are then associated with the Modifier and with the action to do the modification.
  The registered modifications will only occur if the Modifier was passed to 
  the cms.Process' constructor.
  """
  def __init__(self):
    self.__processModifiers = []
    self.__chosen = False
  def makeProcessModifier(self,func):
    """This is used to create a ProcessModifer which can perform actions on the process as a whole.
       This takes as argument a callable object (e.g. function) which takes as its sole argument an instance of Process.
       In order to work, the value returned from this function must be assigned to a uniquely named variable.
    """
    return ProcessModifier(self,func)
  def toModify(self,obj, func=None,**kw):
    """This is used to register an action to be performed on the specific object. Two different forms are allowed
    Form 1: A callable object (e.g. function) can be passed as the second. This callable object is expected to take one argument
    which will be the object passed in as the first argument.
    Form 2: A list of parameter name, value pairs can be passed
       mod.toModify(foo, fred=cms.int32(7), barney = cms.double(3.14))
    """
    if func is not None and len(kw) != 0:
      raise TypeError("toModify takes either two arguments or one argument and key/value pairs")
    if not self.isChosen():
        return
    if func is not None:
      func(obj)
    else:
      temp =_ParameterModifier(kw)
      temp(obj)
  def _setChosen(self):
    """Should only be called by cms.Process instances"""
    self.__chosen = True
  def isChosen(self):
    return self.__chosen

class ModifierChain(object):
    """A Modifier made up of a list of Modifiers
    """
    def __init__(self, *chainedModifiers):
        self.__chosen = False
        self.__chain = chainedModifiers
    def _applyNewProcessModifiers(self,process):
        """Should only be called by cms.Process instances
        applies list of accumulated changes to the process"""
        for m in self.__chain:
            m._applyNewProcessModifiers(process)
    def _setChosen(self):
        """Should only be called by cms.Process instances"""
        self.__chosen = True
        for m in self.__chain:
            m._setChosen()
    def isChosen(self):
        return self.__chosen

class ProcessModifier(object):
    """A class used by a Modifier to affect an entire Process instance.
    When a Process 'loads' a module containing a ProcessModifier, that
    ProcessModifier will be applied to the Process if and only if the 
    Modifier passed to the constructor has been chosen.
    """
    def __init__(self, modifier, func):
        self.__modifier = modifier
        self.__func = func
        self.__seenProcesses = set()
    def apply(self,process):
        if self.__modifier.isChosen():
            if process not in self.__seenProcesses:
                self.__func(process)
                self.__seenProcesses.add(process)

if __name__=="__main__":
    import unittest
    import copy
    
    class TestMakePSet(object):
        """Has same interface as the C++ object which creates PSets
        """
        def __init__(self):
            self.values = dict()
        def __insertValue(self,tracked,label,value):
            self.values[label]=(tracked,value)
        def addInt32(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVInt32(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addUInt32(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVUInt32(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addInt64(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVInt64(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addUInt64(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVUInt64(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addDouble(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVDouble(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addBool(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addString(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVString(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addInputTag(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVInputTag(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addESInputTag(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVESInputTag(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addEventID(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVEventID(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addLuminosityBlockID(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addLuminosityBlockID(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addEventRange(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVEventRange(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addPSet(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVPSet(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addFileInPath(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def newPSet(self):
            return TestMakePSet()
    
    class TestModuleCommand(unittest.TestCase):
        def setUp(self):
            """Nothing to do """
            None
        def testParameterizable(self):
            p = _Parameterizable()
            self.assertEqual(len(p.parameterNames_()),0)
            p.a = int32(1)
            self.assert_('a' in p.parameterNames_())
            self.assertEqual(p.a.value(), 1)
            p.a = 10
            self.assertEqual(p.a.value(), 10)
            p.a = untracked(int32(1))
            self.assertEqual(p.a.value(), 1)
            self.failIf(p.a.isTracked())
            p.a = untracked.int32(1)
            self.assertEqual(p.a.value(), 1)
            self.failIf(p.a.isTracked())
            p = _Parameterizable(foo=int32(10), bar = untracked(double(1.0)))
            self.assertEqual(p.foo.value(), 10)
            self.assertEqual(p.bar.value(),1.0)
            self.failIf(p.bar.isTracked())
            self.assertRaises(TypeError,setattr,(p,'c',1))
            p = _Parameterizable(a=PSet(foo=int32(10), bar = untracked(double(1.0))))
            self.assertEqual(p.a.foo.value(),10)
            self.assertEqual(p.a.bar.value(),1.0)
            p.b = untracked(PSet(fii = int32(1)))
            self.assertEqual(p.b.fii.value(),1)
            self.failIf(p.b.isTracked())
            #test the fact that values can be shared
            v = int32(10)
            p=_Parameterizable(a=v)
            v.setValue(11)
            self.assertEqual(p.a.value(),11)
            p.a = 12
            self.assertEqual(p.a.value(),12)
            self.assertEqual(v.value(),12)
        def testTypedParameterizable(self):
            p = _TypedParameterizable("blah", b=int32(1))
            #see if copy works deeply
            other = p.copy()
            other.b = 2
            self.assertNotEqual(p.b,other.b)

        def testProcessInsertion(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            self.assert_( 'a' in p.analyzers_() )
            self.assert_( 'a' in p.analyzers)
            p.add_(Service("MessageLogger"))
            self.assert_('MessageLogger' in p.services_())
            self.assertEqual(p.MessageLogger.type_(), "MessageLogger")
            p.Tracer = Service("Tracer")
            self.assert_('Tracer' in p.services_())
            self.assertRaises(TypeError, setattr, *(p,'b',"this should fail"))
            self.assertRaises(TypeError, setattr, *(p,'bad',Service("MessageLogger")))
            self.assertRaises(ValueError, setattr, *(p,'bad',Source("PoolSource")))
            p.out = OutputModule("Outer")
            self.assertEqual(p.out.type_(), 'Outer')
            self.assert_( 'out' in p.outputModules_() )

            p.geom = ESSource("GeomProd")
            self.assert_('geom' in p.es_sources_())
            p.add_(ESSource("ConfigDB"))
            self.assert_('ConfigDB' in p.es_sources_())

            p.aliasfoo1 = EDAlias(foo1 = VPSet(PSet(type = string("Foo1"))))
            self.assert_('aliasfoo1' in p.aliases_())

        def testProcessExtend(self):
            class FromArg(object):
                def __init__(self,*arg,**args):
                    for name in args.iterkeys():
                        self.__dict__[name]=args[name]

            a=EDAnalyzer("MyAnalyzer")
            t=EDAnalyzer("MyAnalyzer")
            t.setLabel("foo")
            s1 = Sequence(a)
            s2 = Sequence(s1)
            s3 = Sequence(s2)
            d = FromArg(
                    a=a,
                    b=Service("Full"),
                    c=Path(a),
                    d=s2,
                    e=s1,
                    f=s3,
                    g=Sequence(s1+s2+s3)
                )
            p = Process("Test")
            p.extend(d)
            self.assertEqual(p.a.type_(),"MyAnalyzer")
            self.assertEqual(p.a.label_(),"a")
            self.assertRaises(AttributeError,getattr,p,'b')
            self.assertEqual(p.Full.type_(),"Full")
            self.assertEqual(str(p.c),'a')
            self.assertEqual(str(p.d),'a')

            z1 = FromArg(
                    a=a,
                    b=Service("Full"),
                    c=Path(a),
                    d=s2,
                    e=s1,
                    f=s3,
                    s4=s3,
                    g=Sequence(s1+s2+s3)
                 )
            
            p1 = Process("Test")
            #p1.extend(z1)
            self.assertRaises(ValueError, p1.extend, z1)

            z2 = FromArg(
                    a=a,
                    b=Service("Full"),
                    c=Path(a),
                    d=s2,
                    e=s1,
                    f=s3,
                    aaa=copy.deepcopy(a),
                    s4=copy.deepcopy(s3),
                    g=Sequence(s1+s2+s3),
                    t=t
                )
            p2 = Process("Test")
            p2.extend(z2)
            #self.assertRaises(ValueError, p2.extend, z2)
            self.assertEqual(p2.s4.label_(),"s4")
            #p2.s4.setLabel("foo")
            self.assertRaises(ValueError, p2.s4.setLabel, "foo")
            p2.s4.setLabel("s4")
            p2.s4.setLabel(None)
            p2.s4.setLabel("foo")
            p2._Process__setObjectLabel(p2.s4, "foo")
            p2._Process__setObjectLabel(p2.s4, None)
            p2._Process__setObjectLabel(p2.s4, "bar")

        def testProcessDumpPython(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.p = Path(p.a)
            p.s = Sequence(p.a)
            p.r = Sequence(p.s)
            p.p2 = Path(p.s)
            p.schedule = Schedule(p.p2,p.p)
            d=p.dumpPython()
            self.assertEqual(d,
"""import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.a = cms.EDAnalyzer("MyAnalyzer")


process.s = cms.Sequence(process.a)


process.r = cms.Sequence(process.s)


process.p = cms.Path(process.a)


process.p2 = cms.Path(process.s)


process.schedule = cms.Schedule(*[ process.p2, process.p ])
""")
            #Reverse order of 'r' and 's'
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.p = Path(p.a)
            p.r = Sequence(p.a)
            p.s = Sequence(p.r)
            p.p2 = Path(p.r)
            p.schedule = Schedule(p.p2,p.p)
            p.b = EDAnalyzer("YourAnalyzer")
            d=p.dumpPython()
            self.assertEqual(d,
"""import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.a = cms.EDAnalyzer("MyAnalyzer")


process.b = cms.EDAnalyzer("YourAnalyzer")


process.r = cms.Sequence(process.a)


process.s = cms.Sequence(process.r)


process.p = cms.Path(process.a)


process.p2 = cms.Path(process.r)


process.schedule = cms.Schedule(*[ process.p2, process.p ])
""")
        #use an anonymous sequence
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.p = Path(p.a)
            s = Sequence(p.a)
            p.r = Sequence(s)
            p.p2 = Path(p.r)
            p.schedule = Schedule(p.p2,p.p)
            d=p.dumpPython()
            self.assertEqual(d,
            """import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.a = cms.EDAnalyzer("MyAnalyzer")


process.r = cms.Sequence((process.a))


process.p = cms.Path(process.a)


process.p2 = cms.Path(process.r)


process.schedule = cms.Schedule(*[ process.p2, process.p ])
""")

        def testSecSource(self):
            p = Process('test')
            p.a = SecSource("MySecSource")
            self.assertEqual(p.dumpPython().replace('\n',''),'import FWCore.ParameterSet.Config as cmsprocess = cms.Process("test")process.a = cms.SecSource("MySecSource")')

        def testGlobalReplace(self):
            p = Process('test')
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.s = Sequence(p.a*p.b)
            p.p = Path(p.c+p.s+p.a)
            new = EDAnalyzer("NewAnalyzer")
            p.globalReplace("a",new)

        def testSequence(self):
            p = Process('test')
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.s = Sequence(p.a*p.b)
            self.assertEqual(str(p.s),'a+b')
            self.assertEqual(p.s.label_(),'s')
            path = Path(p.c+p.s)
            self.assertEqual(str(path),'c+a+b')
            p._validateSequence(path, 'p1')
            notInProcess = EDAnalyzer('NotInProcess')
            p2 = Path(p.c+p.s*notInProcess)
            self.assertRaises(RuntimeError, p._validateSequence, p2, 'p2')

        def testSequence2(self):
            p = Process('test')
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            testseq = Sequence(p.a*p.b)
            p.s = testseq
            #p.y = testseq
            self.assertRaises(ValueError, p.__setattr__, "y", testseq) 

        def testPath(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            path = Path(p.a)
            path *= p.b
            path += p.c
            self.assertEqual(str(path),'a+b+c')
            path = Path(p.a*p.b+p.c)
            self.assertEqual(str(path),'a+b+c')
#            path = Path(p.a)*p.b+p.c #This leads to problems with sequences
#            self.assertEqual(str(path),'((a*b)+c)')
            path = Path(p.a+ p.b*p.c)
            self.assertEqual(str(path),'a+b+c')
            path = Path(p.a*(p.b+p.c))
            self.assertEqual(str(path),'a+b+c')
            path = Path(p.a*(p.b+~p.c))
            self.assertEqual(str(path),'a+b+~c')
            p.es = ESProducer("AnESProducer")
            self.assertRaises(TypeError,Path,p.es)

        def testCloneSequence(self):
            p = Process("test")
            a = EDAnalyzer("MyAnalyzer")
            p.a = a
            a.setLabel("a")
            b = EDAnalyzer("YOurAnalyzer")
            p.b = b
            b.setLabel("b")
            path = Path(a * b)
            p.path = Path(p.a*p.b)
            lookuptable = {id(a): p.a, id(b): p.b}
            #self.assertEqual(str(path),str(path._postProcessFixup(lookuptable)))
            #lookuptable = p._cloneToObjectDict
            #self.assertEqual(str(path),str(path._postProcessFixup(lookuptable)))
            self.assertEqual(str(path),str(p.path))

        def testSchedule(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.d = EDAnalyzer("OurAnalyzer")
            p.path1 = Path(p.a)
            p.path2 = Path(p.b)
            p.path3 = Path(p.d)

            s = Schedule(p.path1,p.path2)
            self.assertEqual(s[0],p.path1)
            self.assertEqual(s[1],p.path2)
            p.schedule = s
            self.assert_('b' in p.schedule.moduleNames())
            self.assert_(hasattr(p, 'b'))
            self.assert_(hasattr(p, 'c'))
            self.assert_(hasattr(p, 'd'))
            self.assert_(hasattr(p, 'path1'))
            self.assert_(hasattr(p, 'path2'))
            self.assert_(hasattr(p, 'path3'))
            p.prune()
            self.assert_('b' in p.schedule.moduleNames())
            self.assert_(hasattr(p, 'b'))
            self.assert_(not hasattr(p, 'c'))
            self.assert_(not hasattr(p, 'd'))
            self.assert_(hasattr(p, 'path1'))
            self.assert_(hasattr(p, 'path2'))
            self.assert_(not hasattr(p, 'path3'))

            #adding a path not attached to the Process should cause an exception
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            path1 = Path(p.a)
            s = Schedule(path1)
            self.assertRaises(RuntimeError, lambda : p.setSchedule_(s) )

            #make sure anonymous sequences work
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("MyOtherAnalyzer")
            p.c = EDProducer("MyProd")
            path1 = Path(p.c*Sequence(p.a+p.b))
            s = Schedule(path1)
            self.assert_('a' in s.moduleNames())
            self.assert_('b' in s.moduleNames())
            self.assert_('c' in s.moduleNames())
            p.path1 = path1
            p.schedule = s
            p.prune()
            self.assert_('a' in s.moduleNames())
            self.assert_('b' in s.moduleNames())
            self.assert_('c' in s.moduleNames())

        def testImplicitSchedule(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.path1 = Path(p.a)
            p.path2 = Path(p.b)
            self.assert_(p.schedule is None)
            pths = p.paths
            keys = pths.keys()
            self.assertEqual(pths[keys[0]],p.path1)
            self.assertEqual(pths[keys[1]],p.path2)
            p.prune()
            self.assert_(hasattr(p, 'a'))
            self.assert_(hasattr(p, 'b'))
            self.assert_(not hasattr(p, 'c'))
            self.assert_(hasattr(p, 'path1'))
            self.assert_(hasattr(p, 'path2'))


            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.path2 = Path(p.b)
            p.path1 = Path(p.a)
            self.assert_(p.schedule is None)
            pths = p.paths
            keys = pths.keys()
            self.assertEqual(pths[keys[1]],p.path1)
            self.assertEqual(pths[keys[0]],p.path2)


        def testUsing(self):
            p = Process('test')
            p.block = PSet(a = int32(1))
            p.modu = EDAnalyzer('Analyzer', p.block, b = int32(2))
            self.assertEqual(p.modu.a.value(),1)
            self.assertEqual(p.modu.b.value(),2)

        def testOverride(self):
            p = Process('test')
            a = EDProducer("A", a1=int32(0))
            self.assert_(not a.isModified())
            a.a1 = 1
            self.assert_(a.isModified())
            p.a = a
            self.assertEqual(p.a.a1.value(), 1)
            # try adding an unmodified module.
            # should accept it
            p.a = EDProducer("A", a1=int32(2))
            self.assertEqual(p.a.a1.value(), 2)
            # try adding a modified module.  Should throw
            # no longer, since the same (modified) say, geometry
            # could come from more than one cff
            b = EDProducer("A", a1=int32(3))
            b.a1 = 4
            #self.assertRaises(RuntimeError, setattr, *(p,'a',b))
            ps1 = PSet(a = int32(1))
            ps2 = PSet(a = int32(2))
            self.assertRaises(ValueError, EDProducer, 'C', ps1, ps2)
            self.assertRaises(ValueError, EDProducer, 'C', ps1, a=int32(3))

        def testExamples(self):
            p = Process("Test")
            p.source = Source("PoolSource",fileNames = untracked(string("file:reco.root")))
            p.foos = EDProducer("FooProducer")
            p.bars = EDProducer("BarProducer", foos=InputTag("foos"))
            p.out = OutputModule("PoolOutputModule",fileName=untracked(string("file:foos.root")))
            p.bars.foos = 'Foosball'
            self.assertEqual(p.bars.foos, InputTag('Foosball'))
            p.p = Path(p.foos*p.bars)
            p.e = EndPath(p.out)
            p.add_(Service("MessageLogger"))

        def testPrefers(self):
            p = Process("Test")
            p.add_(ESSource("ForceSource"))
            p.juicer = ESProducer("JuicerProducer")
            p.prefer("ForceSource")
            p.prefer("juicer")
            self.assertEqual(p.dumpConfig(),
"""process Test = {
    es_module juicer = JuicerProducer { 
    }
    es_source  = ForceSource { 
    }
    es_prefer  = ForceSource { 
    }
    es_prefer juicer = JuicerProducer { 
    }
}
""")
            p.prefer("juicer",fooRcd=vstring("Foo"))
            self.assertEqual(p.dumpConfig(),
"""process Test = {
    es_module juicer = JuicerProducer { 
    }
    es_source  = ForceSource { 
    }
    es_prefer  = ForceSource { 
    }
    es_prefer juicer = JuicerProducer { 
        vstring fooRcd = {
            'Foo'
        }

    }
}
""")
            self.assertEqual(p.dumpPython(),
"""import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.juicer = cms.ESProducer("JuicerProducer")


process.ForceSource = cms.ESSource("ForceSource")


process.prefer("ForceSource")

process.prefer("juicer",
    fooRcd = cms.vstring('Foo')
)

""")

        def testFreeze(self):
            process = Process("Freeze")
            m = EDProducer("M", p=PSet(i = int32(1)))
            m.p.i = 2
            process.m = m
            # should be frozen
            #self.assertRaises(ValueError, setattr, m.p, 'i', 3)
            #self.assertRaises(ValueError, setattr, m, 'p', PSet(i=int32(1)))
            #self.assertRaises(ValueError, setattr, m.p, 'j', 1)
            #self.assertRaises(ValueError, setattr, m, 'j', 1)
            # But OK to change through the process
            process.m.p.i = 4
            self.assertEqual(process.m.p.i.value(), 4)
            process.m.p = PSet(j=int32(1))
            # should work to clone it, though
            m2 = m.clone(p = PSet(i = int32(5)), j = int32(8))
            m2.p.i = 6
            m2.j = 8
        def testSubProcess(self):
            process = Process("Parent")
            subProcess = Process("Child")
            subProcess.a = EDProducer("A")
            subProcess.p = Path(subProcess.a)
            subProcess.add_(Service("Foo"))
            process.add_( SubProcess(subProcess) )
            d = process.dumpPython()
            equalD ="""import FWCore.ParameterSet.Config as cms

process = cms.Process("Parent")

parentProcess = process
import FWCore.ParameterSet.Config as cms

process = cms.Process("Child")

process.a = cms.EDProducer("A")


process.p = cms.Path(process.a)


process.Foo = cms.Service("Foo")


childProcess = process
process = parentProcess
process.subProcess = cms.SubProcess( process = childProcess, SelectEvents = cms.untracked.PSet(

), outputCommands = cms.untracked.vstring())
"""
            equalD = equalD.replace("parentProcess","parentProcess"+str(hash(process.subProcess)))
            self.assertEqual(d,equalD)
            p = TestMakePSet()
            process.subProcess.insertInto(p,"dummy")
            self.assertEqual((True,['a']),p.values["@sub_process"][1].values["process"][1].values['@all_modules'])
            self.assertEqual((True,['p']),p.values["@sub_process"][1].values["process"][1].values['@paths'])
            self.assertEqual({'@service_type':(True,'Foo')}, p.values["@sub_process"][1].values["process"][1].values["services"][1][0].values)
        def testRefToPSet(self):
            proc = Process("test")
            proc.top = PSet(a = int32(1))
            proc.ref = PSet(refToPSet_ = string("top"))
            proc.ref2 = PSet( a = int32(1), b = PSet( refToPSet_ = string("top")))
            proc.ref3 = PSet(refToPSet_ = string("ref"))
            proc.ref4 = VPSet(PSet(refToPSet_ = string("top")),
                              PSet(refToPSet_ = string("ref2")))
            p = TestMakePSet()
            proc.fillProcessDesc(p)
            self.assertEqual((True,1),p.values["ref"][1].values["a"])
            self.assertEqual((True,1),p.values["ref3"][1].values["a"])
            self.assertEqual((True,1),p.values["ref2"][1].values["a"])
            self.assertEqual((True,1),p.values["ref2"][1].values["b"][1].values["a"])
            self.assertEqual((True,1),p.values["ref4"][1][0].values["a"])
            self.assertEqual((True,1),p.values["ref4"][1][1].values["a"])
        def testPrune(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.d = EDAnalyzer("OurAnalyzer")
            p.s = Sequence(p.d)
            p.path1 = Path(p.a)
            p.path2 = Path(p.b)
            self.assert_(p.schedule is None)
            pths = p.paths
            keys = pths.keys()
            self.assertEqual(pths[keys[0]],p.path1)
            self.assertEqual(pths[keys[1]],p.path2)
            p.pset1 = PSet(parA = string("pset1"))
            p.pset2 = untracked.PSet(parA = string("pset2"))
            p.vpset1 = VPSet()
            p.vpset2 = untracked.VPSet()
            p.prune()
            self.assert_(hasattr(p, 'a'))
            self.assert_(hasattr(p, 'b'))
            self.assert_(not hasattr(p, 'c'))
            self.assert_(not hasattr(p, 'd'))
            self.assert_(not hasattr(p, 's'))
            self.assert_(hasattr(p, 'path1'))
            self.assert_(hasattr(p, 'path2'))
#            self.assert_(not hasattr(p, 'pset1'))
#            self.assert_(hasattr(p, 'pset2'))
#            self.assert_(not hasattr(p, 'vpset1'))
#            self.assert_(not hasattr(p, 'vpset2'))

            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.d = EDAnalyzer("OurAnalyzer")
            p.e = EDAnalyzer("OurAnalyzer")
            p.s = Sequence(p.d)
            p.s2 = Sequence(p.b)
            p.s3 = Sequence(p.e)
            p.path1 = Path(p.a)
            p.path2 = Path(p.b)
            p.path3 = Path(p.b+p.s2)
            p.path4 = Path(p.b+p.s3)
            p.schedule = Schedule(p.path1,p.path2,p.path3)
            pths = p.paths
            keys = pths.keys()
            self.assertEqual(pths[keys[0]],p.path1)
            self.assertEqual(pths[keys[1]],p.path2)
            p.prune()
            self.assert_(hasattr(p, 'a'))
            self.assert_(hasattr(p, 'b'))
            self.assert_(not hasattr(p, 'c'))
            self.assert_(not hasattr(p, 'd'))
            self.assert_(not hasattr(p, 'e'))
            self.assert_(not hasattr(p, 's'))
            self.assert_(hasattr(p, 's2'))
            self.assert_(not hasattr(p, 's3'))
            self.assert_(hasattr(p, 'path1'))
            self.assert_(hasattr(p, 'path2'))
            self.assert_(hasattr(p, 'path3'))
            self.assert_(not hasattr(p, 'path4'))
            #test SequencePlaceholder
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.s = Sequence(SequencePlaceholder("a")+p.b)
            p.pth = Path(p.s)
            p.prune()
            self.assert_(hasattr(p, 'a'))
            self.assert_(hasattr(p, 'b'))
            self.assert_(hasattr(p, 's'))
            self.assert_(hasattr(p, 'pth'))
        def testModifier(self):
            m1 = Modifier()
            p = Process("test",m1)
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1))
            def _mod_fred(obj):
              obj.fred = 2
            m1.toModify(p.a,_mod_fred)
            self.assertEqual(p.a.fred.value(),2)
            p.b = EDAnalyzer("YourAnalyzer", wilma = int32(1))
            m1.toModify(p.b, wilma = 2)
            self.assertEqual(p.b.wilma.value(),2)
            #check that Modifier not attached to a process doesn't run
            m1 = Modifier()
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1))
            m1.toModify(p.a,_mod_fred)
            p.b = EDAnalyzer("YourAnalyzer", wilma = int32(1))
            m1.toModify(p.b, wilma = 2)
            self.assertEqual(p.a.fred.value(),1)
            self.assertEqual(p.b.wilma.value(),1)
            #make sure clones get the changes
            m1 = Modifier()
            p = Process("test",m1)
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1), wilma = int32(1))
            m1.toModify(p.a, fred = int32(2))
            p.b = p.a.clone(wilma = int32(3))
            self.assertEqual(p.a.fred.value(),2)
            self.assertEqual(p.a.wilma.value(),1)
            self.assertEqual(p.b.fred.value(),2)
            self.assertEqual(p.b.wilma.value(),3)
            #test that load causes process wide methods to run
            def _rem_a(proc):
                del proc.a
            class ProcModifierMod(object):
                def __init__(self,modifier,func):
                    self.proc_mod_ = modifier.makeProcessModifier(func)
            class DummyMod(object):
                def __init__(self):
                    self.a = EDAnalyzer("Dummy")
            testMod = DummyMod()
            p.extend(testMod)
            self.assert_(hasattr(p,"a"))
            m1 = Modifier()
            p = Process("test",m1)
            testProcMod = ProcModifierMod(m1,_rem_a)
            p.extend(testMod)
            p.extend(testProcMod)
            self.assert_(not hasattr(p,"a"))
            #test ModifierChain
            m1 = Modifier()
            mc = ModifierChain(m1)
            p = Process("test",mc)
            testMod = DummyMod()
            p.b = EDAnalyzer("Dummy2", fred = int32(1))
            m1.toModify(p.b, fred = int32(3))
            p.extend(testMod)
            testProcMod = ProcModifierMod(m1,_rem_a)
            p.extend(testProcMod)
            self.assert_(not hasattr(p,"a"))
            self.assertEqual(p.b.fred.value(),3)


    unittest.main()
