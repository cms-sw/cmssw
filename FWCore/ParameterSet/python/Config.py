#!/usr/bin/env python

### command line options helper
from __future__ import print_function
from __future__ import absolute_import
import six
import os
from  .Options import Options
options = Options()


## imports
import sys
from .Mixins import PrintOptions,_ParameterTypeBase,_SimpleParameterTypeBase, _Parameterizable, _ConfigureComponent, _TypedParameterizable, _Labelable,  _Unlabelable,  _ValidatingListBase, _modifyParametersFromDict
from .Mixins import *
from .Types import *
from .Modules import *
from .Modules import _Module
from .SequenceTypes import *
from .SequenceTypes import _ModuleSequenceType, _Sequenceable  #extend needs it
from .SequenceVisitors import PathValidator, EndPathValidator, ScheduleTaskValidator, NodeVisitor, CompositeVisitor, ModuleNamesFromGlobalsVisitor
from . import DictTypes

from .ExceptionHandling import *

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

    ignorePatterns = ['FWCore/ParameterSet/Config.py','<string>','<frozen ']
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
            that will be used to modify the Process as it is built
            """
        self.__dict__['_Process__name'] = name
        if not name.isalnum():
            raise RuntimeError("Error: The process name is an empty string or contains non-alphanumeric characters")
        self.__dict__['_Process__filters'] = {}
        self.__dict__['_Process__producers'] = {}
        self.__dict__['_Process__switchproducers'] = {}
        self.__dict__['_Process__source'] = None
        self.__dict__['_Process__looper'] = None
        self.__dict__['_Process__subProcesses'] = []
        self.__dict__['_Process__schedule'] = None
        self.__dict__['_Process__analyzers'] = {}
        self.__dict__['_Process__outputmodules'] = {}
        self.__dict__['_Process__paths'] = DictTypes.SortedKeysDict()    # have to keep the order
        self.__dict__['_Process__endpaths'] = DictTypes.SortedKeysDict() # of definition
        self.__dict__['_Process__sequences'] = {}
        self.__dict__['_Process__tasks'] = {}
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
        self.options = Process.defaultOptions_()
        self.maxEvents = Process.defaultMaxEvents_()
        self.maxLuminosityBlocks = Process.defaultMaxLuminosityBlocks_()
        for m in self.__modifiers:
            m._setChosen()

    def setStrict(self, value):
        self.__isStrict = value
        _Module.__isStrict__ = True

    # some user-friendly methods for command-line browsing
    def producerNames(self):
        """Returns a string containing all the EDProducer labels separated by a blank"""
        return ' '.join(self.producers_().keys())
    def switchProducerNames(self):
        """Returns a string containing all the SwitchProducer labels separated by a blank"""
        return ' '.join(self.switchProducers_().keys())
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
        """returns a dict of the filters that have been added to the Process"""
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
        """returns a dict of the producers that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__producers)
    producers = property(producers_,doc="dictionary containing the producers for the process")
    def switchProducers_(self):
        """returns a dict of the SwitchProducers that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__switchproducers)
    switchProducers = property(switchProducers_,doc="dictionary containing the SwitchProducers for the process")
    def source_(self):
        """returns the source that has been added to the Process or None if none have been added"""
        return self.__source
    def setSource_(self,src):
        self._placeSource('source',src)
    source = property(source_,setSource_,doc='the main source or None if not set')
    def looper_(self):
        """returns the looper that has been added to the Process or None if none have been added"""
        return self.__looper
    def setLooper_(self,lpr):
        self._placeLooper('looper',lpr)
    looper = property(looper_,setLooper_,doc='the main looper or None if not set')
    @staticmethod
    def defaultOptions_():
        return untracked.PSet(numberOfThreads = untracked.uint32(1),
                              numberOfStreams = untracked.uint32(0),
                              numberOfConcurrentRuns = untracked.uint32(1),
                              numberOfConcurrentLuminosityBlocks = untracked.uint32(1),
                              eventSetup = untracked.PSet(
                                  numberOfConcurrentIOVs = untracked.uint32(1),
                                  forceNumberOfConcurrentIOVs = untracked.PSet(
                                      allowAnyLabel_ = required.untracked.uint32
                                  )
                              ),
                              wantSummary = untracked.bool(False),
                              fileMode = untracked.string('FULLMERGE'),
                              forceEventSetupCacheClearOnNewRun = untracked.bool(False),
                              throwIfIllegalParameter = untracked.bool(True),
                              printDependencies = untracked.bool(False),
                              deleteNonConsumedUnscheduledModules = untracked.bool(True),
                              sizeOfStackForThreadsInKB = optional.untracked.uint32,
                              Rethrow = untracked.vstring(),
                              SkipEvent = untracked.vstring(),
                              FailPath = untracked.vstring(),
                              IgnoreCompletely = untracked.vstring(),
                              canDeleteEarly = untracked.vstring(),
                              allowUnscheduled = obsolete.untracked.bool,
                              emptyRunLumiMode = obsolete.untracked.string,
                              makeTriggerResults = obsolete.untracked.bool
                              )
    def __updateOptions(self,opt):
        newOpts = self.defaultOptions_()
        if isinstance(opt,dict):
            for k,v in six.iteritems(opt):
                setattr(newOpts,k,v)
        else:
            for p in opt.parameters_():
                setattr(newOpts, p, getattr(opt,p))
        return newOpts
    @staticmethod
    def defaultMaxEvents_():
        return untracked.PSet(input=optional.untracked.int32,
                              output=optional.untracked.allowed(int32,PSet))
    def __updateMaxEvents(self,ps):
        newMax = self.defaultMaxEvents_()
        if isinstance(ps,dict):
            for k,v in six.iteritems(ps):
                setattr(newMax,k,v)
        else:
            for p in ps.parameters_():
                setattr(newMax, p, getattr(ps,p))
        return newMax
    @staticmethod
    def defaultMaxLuminosityBlocks_():
        return untracked.PSet(input=untracked.int32(-1))
    def subProcesses_(self):
        """returns a list of the subProcesses that have been added to the Process"""
        return self.__subProcesses
    subProcesses = property(subProcesses_,doc='the SubProcesses that have been added to the Process')
    def analyzers_(self):
        """returns a dict of the analyzers that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__analyzers)
    analyzers = property(analyzers_,doc="dictionary containing the analyzers for the process")
    def outputModules_(self):
        """returns a dict of the output modules that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__outputmodules)
    outputModules = property(outputModules_,doc="dictionary containing the output_modules for the process")
    def paths_(self):
        """returns a dict of the paths that have been added to the Process"""
        return DictTypes.SortedAndFixedKeysDict(self.__paths)
    paths = property(paths_,doc="dictionary containing the paths for the process")
    def endpaths_(self):
        """returns a dict of the endpaths that have been added to the Process"""
        return DictTypes.SortedAndFixedKeysDict(self.__endpaths)
    endpaths = property(endpaths_,doc="dictionary containing the endpaths for the process")
    def sequences_(self):
        """returns a dict of the sequences that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__sequences)
    sequences = property(sequences_,doc="dictionary containing the sequences for the process")
    def tasks_(self):
        """returns a dict of the tasks that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__tasks)
    tasks = property(tasks_,doc="dictionary containing the tasks for the process")
    def schedule_(self):
        """returns the schedule that has been added to the Process or None if none have been added"""
        return self.__schedule
    def setPartialSchedule_(self,sch,label):
        if label == "schedule":
            self.setSchedule_(sch)
        else:
            self._place(label, sch, self.__partialschedules)
    def setSchedule_(self,sch):
        # See if every path and endpath has been inserted into the process
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
        """returns a dict of the services that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__services)
    services = property(services_,doc="dictionary containing the services for the process")
    def es_producers_(self):
        """returns a dict of the esproducers that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__esproducers)
    es_producers = property(es_producers_,doc="dictionary containing the es_producers for the process")
    def es_sources_(self):
        """returns a the es_sources that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__essources)
    es_sources = property(es_sources_,doc="dictionary containing the es_sources for the process")
    def es_prefers_(self):
        """returns a dict of the es_prefers that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__esprefers)
    es_prefers = property(es_prefers_,doc="dictionary containing the es_prefers for the process")
    def aliases_(self):
        """returns a dict of the aliases that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__aliases)
    aliases = property(aliases_,doc="dictionary containing the aliases for the process")
    def psets_(self):
        """returns a dict of the PSets that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__psets)
    psets = property(psets_,doc="dictionary containing the PSets for the process")
    def vpsets_(self):
        """returns a dict of the VPSets that have been added to the Process"""
        return DictTypes.FixedKeysDict(self.__vpsets)
    vpsets = property(vpsets_,doc="dictionary containing the PSets for the process")

    def isUsingModifier(self,mod):
        """returns True if the Modifier is in used by this Process"""
        if mod._isChosen():
            for m in self.__modifiers:
                if m._isOrContains(mod):
                    return True
        return False

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

        if name == 'options':
            value = self.__updateOptions(value)
        if name == 'maxEvents':
            value = self.__updateMaxEvents(value)

        # private variable exempt from all this
        if name.startswith('_Process__'):
            self.__dict__[name]=value
            return
        if not isinstance(value,_ConfigureComponent):
            raise TypeError("can only assign labels to an object that inherits from '_ConfigureComponent'\n"
                            +"an instance of "+str(type(value))+" will not work - requested label is "+name)
        if not isinstance(value,_Labelable) and not isinstance(value,Source) and not isinstance(value,Looper) and not isinstance(value,Schedule):
            if name == value.type_():
                # Only Services get handled here
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
            # Complain if items in sequences or tasks from load() statements have
            # degenerate names, but if the user overwrites a name in the
            # main config, replace it everywhere
            if newValue._isTaskComponent():
                if not self.__InExtendCall:
                    self._replaceInTasks(name, newValue)
                    self._replaceInSchedule(name, newValue)
                else:
                    if not isinstance(newValue, Task):
                        #should check to see if used in task before complaining
                        newFile='top level config'
                        if hasattr(value,'_filename'):
                            newFile = value._filename
                        oldFile='top level config'
                        oldValue = getattr(self,name)
                        if hasattr(oldValue,'_filename'):
                            oldFile = oldValue._filename
                        msg1 = "Trying to override definition of "+name+" while it is used by the task "
                        msg2 = "\n new object defined in: "+newFile
                        msg2 += "\n existing object defined in: "+oldFile
                        s = self.__findFirstUsingModule(self.tasks,oldValue)
                        if s is not None:
                            raise ValueError(msg1+s.label_()+msg2)

            if isinstance(newValue, _Sequenceable) or newValue._isTaskComponent():
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
                    msg1 = "Trying to override definition of "+name+" while it is used by the "
                    msg2 = "\n new object defined in: "+newFile
                    msg2 += "\n existing object defined in: "+oldFile
                    s = self.__findFirstUsingModule(self.sequences,oldValue)
                    if s is not None:
                        raise ValueError(msg1+"sequence "+s.label_()+msg2)
                    s = self.__findFirstUsingModule(self.paths,oldValue)
                    if s is not None:
                        raise ValueError(msg1+"path "+s.label_()+msg2)
                    s = self.__findFirstUsingModule(self.endpaths,oldValue)
                    if s is not None:
                        raise ValueError(msg1+"endpath "+s.label_()+msg2)

            # In case of EDAlias, raise Exception always to avoid surprises
            if isinstance(newValue, EDAlias):
                oldValue = getattr(self, name)
                #should check to see if used in task/sequence before complaining
                newFile='top level config'
                if hasattr(value,'_filename'):
                    newFile = value._filename
                oldFile='top level config'
                if hasattr(oldValue,'_filename'):
                    oldFile = oldValue._filename
                msg1 = "Trying to override definition of "+name+" with an EDAlias while it is used by the "
                msg2 = "\n new object defined in: "+newFile
                msg2 += "\n existing object defined in: "+oldFile
                s = self.__findFirstUsingModule(self.tasks,oldValue)
                if s is not None:
                    raise ValueError(msg1+"task "+s.label_()+msg2)
                s = self.__findFirstUsingModule(self.sequences,oldValue)
                if s is not None:
                    raise ValueError(msg1+"sequence "+s.label_()+msg2)
                s = self.__findFirstUsingModule(self.paths,oldValue)
                if s is not None:
                    raise ValueError(msg1+"path "+s.label_()+msg2)
                s = self.__findFirstUsingModule(self.endpaths,oldValue)
                if s is not None:
                    raise ValueError(msg1+"endpath "+s.label_()+msg2)

            self._delattrFromSetattr(name)
        self.__dict__[name]=newValue
        if isinstance(newValue,_Labelable):
            self.__setObjectLabel(newValue, name)
            self._cloneToObjectDict[id(value)] = newValue
            self._cloneToObjectDict[id(newValue)] = newValue
        #now put in proper bucket
        newValue._place(name,self)
    def __findFirstUsingModule(self, seqsOrTasks, mod):
        """Given a container of sequences or tasks, find the first sequence or task
        containing mod and return it. If none is found, return None"""
        from FWCore.ParameterSet.SequenceTypes import ModuleNodeVisitor
        l = list()
        for seqOrTask in six.itervalues(seqsOrTasks):
            l[:] = []
            v = ModuleNodeVisitor(l)
            seqOrTask.visit(v)
            if mod in l:
                return seqOrTask
        return None

    def _delHelper(self,name):
        if not hasattr(self,name):
            raise KeyError('process does not know about '+name)
        elif name.startswith('_Process__'):
            raise ValueError('this attribute cannot be deleted')

        # we have to remove it from all dictionaries/registries
        dicts = [item for item in self.__dict__.values() if (isinstance(item, dict) or isinstance(item, DictTypes.SortedKeysDict))]
        for reg in dicts:
            if name in reg: del reg[name]
        # if it was a labelable object, the label needs to be removed
        obj = getattr(self,name)
        if isinstance(obj,_Labelable):
            obj.setLabel(None)
        if isinstance(obj,Service):
            obj._inProcess = False

    def __delattr__(self,name):
        self._delHelper(name)
        obj = getattr(self,name)
        if not obj is None:
            if not isinstance(obj, Sequence) and not isinstance(obj, Task):
                # For modules, ES modules and services we can also remove
                # the deleted object from Sequences, Paths, EndPaths, and
                # Tasks. Note that for Sequences and Tasks that cannot be done
                # reliably as the places where the Sequence or Task was used
                # might have been expanded so we do not even try. We considered
                # raising an exception if a Sequences or Task was explicitly
                # deleted, but did not because when done carefully deletion
                # is sometimes OK (for example in the prune function where it
                # has been checked that the deleted Sequence is not used).
                if obj._isTaskComponent():
                    self._replaceInTasks(name, None)
                    self._replaceInSchedule(name, None)
                if isinstance(obj, _Sequenceable) or obj._isTaskComponent():
                    self._replaceInSequences(name, None)
        # now remove it from the process itself
        try:
            del self.__dict__[name]
        except:
            pass

    def _delattrFromSetattr(self,name):
        """Similar to __delattr__ but we need different behavior when called from __setattr__"""
        self._delHelper(name)
        # now remove it from the process itself
        try:
            del self.__dict__[name]
        except:
            pass

    def add_(self,value):
        """Allows addition of components that do not have to have a label, e.g. Services"""
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
    def _placeSwitchProducer(self,name,mod):
        self._place(name, mod, self.__switchproducers)
    def _placeFilter(self,name,mod):
        self._place(name, mod, self.__filters)
    def _placeAnalyzer(self,name,mod):
        self._place(name, mod, self.__analyzers)
    def _placePath(self,name,mod):
        self._validateSequence(mod, name)
        try:
            self._place(name, mod, self.__paths)
        except ModuleCloneError as msg:
            context = format_outerframe(4)
            raise Exception("%sThe module %s in path %s is unknown to the process %s." %(context, msg, name, self._Process__name))
    def _placeEndPath(self,name,mod):
        self._validateSequence(mod, name)
        try:
            self._place(name, mod, self.__endpaths)
        except ModuleCloneError as msg:
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
    def _placeTask(self,name,task):
        self._validateTask(task, name)
        self._place(name, task, self.__tasks)
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
        self.__dict__['_Process__subProcess'] = mod
        self.__dict__[mod.type_()] = mod
    def addSubProcess(self,mod):
        self.__subProcesses.append(mod)
    def _placeService(self,typeName,mod):
        self._place(typeName, mod, self.__services)
        if typeName in self.__dict__:
            self.__dict__[typeName]._inProcess = False
        self.__dict__[typeName]=mod
    def load(self, moduleName):
        moduleName = moduleName.replace("/",".")
        module = __import__(moduleName)
        self.extend(sys.modules[moduleName])
    def extend(self,other,items=()):
        """Look in other and find types that we can use"""
        # enable explicit check to avoid overwriting of existing objects
        self.__dict__['_Process__InExtendCall'] = True

        seqs = dict()
        tasksToAttach = dict()
        mods = []
        for name in dir(other):
            #'from XX import *' ignores these, and so should we.
            if name.startswith('_'):
                continue
            item = getattr(other,name)
            if name == "source" or name == "looper":
                # In these cases 'item' could be None if the specific object was not defined
                if item is not None:
                    self.__setattr__(name,item)
            elif isinstance(item,_ModuleSequenceType):
                seqs[name]=item
            elif isinstance(item,Task):
                tasksToAttach[name] = item
            elif isinstance(item,_Labelable):
                self.__setattr__(name,item)
                if not item.hasLabel_() :
                    item.setLabel(name)
            elif isinstance(item,Schedule):
                self.__setattr__(name,item)
            elif isinstance(item,_Unlabelable):
                self.add_(item)
            elif isinstance(item,ProcessModifier):
                mods.append(item)
            elif isinstance(item,ProcessFragment):
                self.extend(item)

        #now create a sequence that uses the newly made items
        for name,seq in six.iteritems(seqs):
            if id(seq) not in self._cloneToObjectDict:
                self.__setattr__(name,seq)
            else:
                newSeq = self._cloneToObjectDict[id(seq)]
                self.__dict__[name]=newSeq
                self.__setObjectLabel(newSeq, name)
                #now put in proper bucket
                newSeq._place(name,self)

        for name, task in six.iteritems(tasksToAttach):
            self.__setattr__(name, task)

        #apply modifiers now that all names have been added
        for item in mods:
            item.apply(self)

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

        config+=self._dumpConfigNamedList(self.subProcesses_(),
                                  'subProcess',
                                  options)
        config+=self._dumpConfigNamedList(six.iteritems(self.producers_()),
                                  'module',
                                  options)
        config+=self._dumpConfigNamedList(six.iteritems(self.switchProducers_()),
                                  'module',
                                  options)
        config+=self._dumpConfigNamedList(six.iteritems(self.filters_()),
                                  'module',
                                  options)
        config+=self._dumpConfigNamedList(six.iteritems(self.analyzers_()),
                                  'module',
                                  options)
        config+=self._dumpConfigNamedList(six.iteritems(self.outputModules_()),
                                  'module',
                                  options)
        config+=self._dumpConfigNamedList(six.iteritems(self.sequences_()),
                                  'sequence',
                                  options)
        config+=self._dumpConfigNamedList(six.iteritems(self.paths_()),
                                  'path',
                                  options)
        config+=self._dumpConfigNamedList(six.iteritems(self.endpaths_()),
                                  'endpath',
                                  options)
        config+=self._dumpConfigUnnamedList(six.iteritems(self.services_()),
                                  'service',
                                  options)
        config+=self._dumpConfigNamedList(six.iteritems(self.aliases_()),
                                  'alias',
                                  options)
        config+=self._dumpConfigOptionallyNamedList(
            six.iteritems(self.es_producers_()),
            'es_module',
            options)
        config+=self._dumpConfigOptionallyNamedList(
            six.iteritems(self.es_sources_()),
            'es_source',
            options)
        config += self._dumpConfigESPrefers(options)
        for name,item in six.iteritems(self.psets):
            config +=options.indentation()+item.configTypeName()+' '+name+' = '+item.configValue(options)
        for name,item in six.iteritems(self.vpsets):
            config +=options.indentation()+'VPSet '+name+' = '+item.configValue(options)
        if self.schedule:
            pathNames = [p.label_() for p in self.schedule]
            config +=options.indentation()+'schedule = {'+','.join(pathNames)+'}\n'

#        config+=self._dumpConfigNamedList(six.iteritems(self.vpsets),
#                                  'VPSet',
#                                  options)
        config += "}\n"
        options.unindent()
        return config

    def _dumpConfigESPrefers(self, options):
        result = ''
        for item in six.itervalues(self.es_prefers_()):
            result +=options.indentation()+'es_prefer '+item.targetLabel_()+' = '+item.dumpConfig(options)
        return result

    def _dumpPythonSubProcesses(self, l, options):
        returnValue = ''
        for item in l:
            returnValue += item.dumpPython(options)+'\n\n'
        return returnValue

    def _dumpPythonList(self, d, options):
        returnValue = ''
        if isinstance(d, DictTypes.SortedKeysDict):
            for name,item in d.items():
                returnValue +='process.'+name+' = '+item.dumpPython(options)+'\n\n'
        else:
            for name,item in sorted(d.items()):
                returnValue +='process.'+name+' = '+item.dumpPython(options)+'\n\n'
        return returnValue

    def _splitPythonList(self, subfolder, d, options):
        parts = DictTypes.SortedKeysDict()
        for name, item in d.items() if isinstance(d, DictTypes.SortedKeysDict) else sorted(d.items()):
            code = ''
            dependencies = item.directDependencies()
            for module_subfolder, module in dependencies:
                module = module + '_cfi'
                if options.useSubdirectories and module_subfolder:
                    module = module_subfolder + '.' + module
                if options.targetDirectory is not None:
                    if options.useSubdirectories and subfolder:
                      module = '..' + module
                    else:
                      module = '.' + module
                code += 'from ' + module + ' import *\n'
            if dependencies:
                code += '\n'
            code += name + ' = ' + item.dumpPython(options)
            parts[name] = subfolder, code
        return parts

    def _validateSequence(self, sequence, label):
        # See if every module has been inserted into the process
        try:
            l = set()
            visitor = NodeNameVisitor(l)
            sequence.visit(visitor)
        except:
            raise RuntimeError("An entry in sequence "+label + ' has no label')

    def _validateTask(self, task, label):
        # See if every module and service has been inserted into the process
        try:
            l = set()
            visitor = NodeNameVisitor(l)
            task.visit(visitor)
        except:
            raise RuntimeError("An entry in task " + label + ' has not been attached to the process')

    def _itemsInDependencyOrder(self, processDictionaryOfItems):
        # The items can be Sequences or Tasks and the input
        # argument should either be the dictionary of sequences
        # or the dictionary of tasks from the process.

        returnValue=DictTypes.SortedKeysDict()

        # For each item, see what other items it depends upon
        # For our purpose here, an item depends on the items it contains.
        dependencies = {}
        for label,item in six.iteritems(processDictionaryOfItems):
            containedItems = []
            if isinstance(item, Task):
                v = TaskVisitor(containedItems)
            else:
                v = SequenceVisitor(containedItems)
            try:
                item.visit(v)
            except RuntimeError:
                if isinstance(item, Task):
                    raise RuntimeError("Failed in a Task visitor. Probably " \
                                       "a circular dependency discovered in Task with label " + label)
                else:
                    raise RuntimeError("Failed in a Sequence visitor. Probably a " \
                                       "circular dependency discovered in Sequence with label " + label)
            for containedItem in containedItems:
                # Check for items that both have labels and are not in the process.
                # This should not normally occur unless someone explicitly assigns a
                # label without putting the item in the process (which should not ever
                # be done). We check here because this problem could cause the code
                # in the 'while' loop below to go into an infinite loop.
                if containedItem.hasLabel_():
                    testItem = processDictionaryOfItems.get(containedItem.label_())
                    if testItem is None or containedItem != testItem:
                        if isinstance(item, Task):
                            raise RuntimeError("Task has a label, but using its label to get an attribute" \
                                               " from the process yields a different object or None\n"+
                                               "label = " + containedItem.label_())
                        else:
                            raise RuntimeError("Sequence has a label, but using its label to get an attribute" \
                                               " from the process yields a different object or None\n"+
                                               "label = " + containedItem.label_())
            dependencies[label]=[dep.label_() for dep in containedItems if dep.hasLabel_()]

        # keep looping until we get rid of all dependencies
        while dependencies:
            oldDeps = dict(dependencies)
            for label,deps in six.iteritems(oldDeps):
                if len(deps)==0:
                    returnValue[label]=processDictionaryOfItems[label]
                    #remove this as a dependency for all other tasks
                    del dependencies[label]
                    for lb2,deps2 in six.iteritems(dependencies):
                        while deps2.count(label):
                            deps2.remove(label)
        return returnValue

    def _dumpPython(self, d, options):
        result = ''
        for name, value in sorted(six.iteritems(d)):
            result += value.dumpPythonAs(name,options)+'\n'
        return result

    def _splitPython(self, subfolder, d, options):
        result = {}
        for name, value in sorted(six.iteritems(d)):
            result[name] = subfolder, value.dumpPythonAs(name, options) + '\n'
        return result

    def dumpPython(self, options=PrintOptions()):
        """return a string containing the equivalent process defined using python"""
        specialImportRegistry._reset()
        header = "import FWCore.ParameterSet.Config as cms"
        result = "process = cms.Process(\""+self.__name+"\")\n\n"
        if self.source_():
            result += "process.source = "+self.source_().dumpPython(options)
        if self.looper_():
            result += "process.looper = "+self.looper_().dumpPython()
        result+=self._dumpPythonList(self.psets, options)
        result+=self._dumpPythonList(self.vpsets, options)
        result+=self._dumpPythonSubProcesses(self.subProcesses_(), options)
        result+=self._dumpPythonList(self.producers_(), options)
        result+=self._dumpPythonList(self.switchProducers_(), options)
        result+=self._dumpPythonList(self.filters_() , options)
        result+=self._dumpPythonList(self.analyzers_(), options)
        result+=self._dumpPythonList(self.outputModules_(), options)
        result+=self._dumpPythonList(self.services_(), options)
        result+=self._dumpPythonList(self.es_producers_(), options)
        result+=self._dumpPythonList(self.es_sources_(), options)
        result+=self._dumpPython(self.es_prefers_(), options)
        result+=self._dumpPythonList(self._itemsInDependencyOrder(self.tasks), options)
        result+=self._dumpPythonList(self._itemsInDependencyOrder(self.sequences), options)
        result+=self._dumpPythonList(self.paths_(), options)
        result+=self._dumpPythonList(self.endpaths_(), options)
        result+=self._dumpPythonList(self.aliases_(), options)
        if not self.schedule_() == None:
            result += 'process.schedule = ' + self.schedule.dumpPython(options)
        imports = specialImportRegistry.getSpecialImports()
        if len(imports) > 0:
            header += "\n" + "\n".join(imports)
        header += "\n\n"
        return header+result

    def splitPython(self, options = PrintOptions()):
        """return a map of file names to python configuration fragments"""
        specialImportRegistry._reset()
        # extract individual fragments
        options.isCfg = False
        header = "import FWCore.ParameterSet.Config as cms"
        result = ''
        parts = {}
        files = {}

        result = 'process = cms.Process("' + self.__name + '")\n\n'

        if self.source_():
            parts['source'] = (None, 'source = ' + self.source_().dumpPython(options))

        if self.looper_():
            parts['looper'] = (None, 'looper = ' + self.looper_().dumpPython())

        parts.update(self._splitPythonList('psets', self.psets, options))
        parts.update(self._splitPythonList('psets', self.vpsets, options))
        # FIXME
        #parts.update(self._splitPythonSubProcesses(self.subProcesses_(), options))
        if len(self.subProcesses_()):
          sys.stderr.write("error: subprocesses are not supported yet\n\n")
        parts.update(self._splitPythonList('modules', self.producers_(), options))
        parts.update(self._splitPythonList('modules', self.switchProducers_(), options))
        parts.update(self._splitPythonList('modules', self.filters_() , options))
        parts.update(self._splitPythonList('modules', self.analyzers_(), options))
        parts.update(self._splitPythonList('modules', self.outputModules_(), options))
        parts.update(self._splitPythonList('services', self.services_(), options))
        parts.update(self._splitPythonList('eventsetup', self.es_producers_(), options))
        parts.update(self._splitPythonList('eventsetup', self.es_sources_(), options))
        parts.update(self._splitPython('eventsetup', self.es_prefers_(), options))
        parts.update(self._splitPythonList('tasks', self._itemsInDependencyOrder(self.tasks), options))
        parts.update(self._splitPythonList('sequences', self._itemsInDependencyOrder(self.sequences), options))
        parts.update(self._splitPythonList('paths', self.paths_(), options))
        parts.update(self._splitPythonList('paths', self.endpaths_(), options))
        parts.update(self._splitPythonList('modules', self.aliases_(), options))

        if options.targetDirectory is not None:
            files[options.targetDirectory + '/__init__.py'] = ''

        if options.useSubdirectories:
          for sub in 'psets', 'modules', 'services', 'eventsetup', 'tasks', 'sequences', 'paths':
            if options.targetDirectory is not None:
                sub = options.targetDirectory + '/' + sub
            files[sub + '/__init__.py'] = ''

        for (name, (subfolder, code)) in six.iteritems(parts):
            filename = name + '_cfi'
            if options.useSubdirectories and subfolder:
                filename = subfolder + '/' + filename
            if options.targetDirectory is not None:
                filename = options.targetDirectory + '/' + filename
            result += 'process.load("%s")\n' % filename
            files[filename + '.py'] = header + '\n\n' + code

        if self.schedule_() is not None:
            options.isCfg = True
            result += 'process.schedule = ' + self.schedule.dumpPython(options)

        imports = specialImportRegistry.getSpecialImports()
        if len(imports) > 0:
            header += '\n' + '\n'.join(imports)
        files['-'] = header + '\n\n' + result
        return files

    def _replaceInSequences(self, label, new):
        old = getattr(self,label)
        #TODO - replace by iterator concatenation
        #to ovoid dependency problems between sequences, first modify
        # process known sequences to do a non-recursive change. Then do
        # a recursive change to get cases where a sub-sequence unknown to
        # the process has the item to be replaced
        for sequenceable in six.itervalues(self.sequences):
            sequenceable._replaceIfHeldDirectly(old,new)
        for sequenceable in six.itervalues(self.sequences):
            sequenceable.replace(old,new)
        for sequenceable in six.itervalues(self.paths):
            sequenceable.replace(old,new)
        for sequenceable in six.itervalues(self.endpaths):
            sequenceable.replace(old,new)
    def _replaceInTasks(self, label, new):
        old = getattr(self,label)
        for task in six.itervalues(self.tasks):
            task.replace(old, new)
    def _replaceInSchedule(self, label, new):
        if self.schedule_() == None:
            return
        old = getattr(self,label)
        for task in self.schedule_()._tasks:
            task.replace(old, new)
    def globalReplace(self,label,new):
        """ Replace the item with label 'label' by object 'new' in the process and all sequences/paths/tasks"""
        if not hasattr(self,label):
            raise LookupError("process has no item of label "+label)
        setattr(self,label,new)
    def _insertInto(self, parameterSet, itemDict):
        for name,value in six.iteritems(itemDict):
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
        for name,value in six.iteritems(itemDict):
            value.appendToProcessDescList_(l, name)
            value.insertInto(parameterSet, name)
        # alphabetical order is easier to compare with old language
        l.sort()
        parameterSet.addVString(tracked, label, l)
    def _insertSwitchProducersInto(self, parameterSet, labelModules, labelAliases, itemDict, tracked):
        modules = parameterSet.getVString(tracked, labelModules)
        aliases = parameterSet.getVString(tracked, labelAliases)
        for name,value in six.iteritems(itemDict):
            value.appendToProcessDescLists_(modules, aliases, name)
            value.insertInto(parameterSet, name)
        modules.sort()
        aliases.sort()
        parameterSet.addVString(tracked, labelModules, modules)
        parameterSet.addVString(tracked, labelAliases, aliases)
    def _insertSubProcessesInto(self, parameterSet, label, itemList, tracked):
        l = []
        subprocs = []
        for value in itemList:
            name = value.getProcessName()
            newLabel = value.nameInProcessDesc_(name)
            l.append(newLabel)
            pset = value.getSubProcessPSet(parameterSet)
            subprocs.append(pset)
        # alphabetical order is easier to compare with old language
        l.sort()
        parameterSet.addVString(tracked, label, l)
        parameterSet.addVPSet(False,"subProcesses",subprocs)
    def _insertPaths(self, processPSet, nodeVisitor):
        scheduledPaths = []
        triggerPaths = []
        endpaths = []
        if self.schedule_() == None:
            # make one from triggerpaths & endpaths
            for name in self.paths_():
                scheduledPaths.append(name)
                triggerPaths.append(name)
            for name in self.endpaths_():
                scheduledPaths.append(name)
                endpaths.append(name)
        else:
            for path in self.schedule_():
                pathname = path.label_()
                scheduledPaths.append(pathname)
                if pathname in self.endpaths_():
                    endpaths.append(pathname)
                else:
                    triggerPaths.append(pathname)
            for task in self.schedule_()._tasks:
                task.resolve(self.__dict__)
                scheduleTaskValidator = ScheduleTaskValidator()
                task.visit(scheduleTaskValidator)
                task.visit(nodeVisitor)
        processPSet.addVString(True, "@end_paths", endpaths)
        processPSet.addVString(True, "@paths", scheduledPaths)
        # trigger_paths are a little different
        p = processPSet.newPSet()
        p.addVString(True, "@trigger_paths", triggerPaths)
        processPSet.addPSet(True, "@trigger_paths", p)
        # add all these paths
        pathValidator = PathValidator()
        endpathValidator = EndPathValidator()
        decoratedList = []
        lister = DecoratedNodeNameVisitor(decoratedList)
        pathCompositeVisitor = CompositeVisitor(pathValidator, nodeVisitor, lister)
        endpathCompositeVisitor = CompositeVisitor(endpathValidator, nodeVisitor, lister)
        for triggername in triggerPaths:
            iPath = self.paths_()[triggername]
            iPath.resolve(self.__dict__)
            pathValidator.setLabel(triggername)
            lister.initialize()
            iPath.visit(pathCompositeVisitor)
            iPath.insertInto(processPSet, triggername, decoratedList)
        for endpathname in endpaths:
            iEndPath = self.endpaths_()[endpathname]
            iEndPath.resolve(self.__dict__)
            endpathValidator.setLabel(endpathname)
            lister.initialize()
            iEndPath.visit(endpathCompositeVisitor)
            iEndPath.insertInto(processPSet, endpathname, decoratedList)
        processPSet.addVString(False, "@filters_on_endpaths", endpathValidator.filtersOnEndpaths)

    def resolve(self,keepUnresolvedSequencePlaceholders=False):
        for x in six.itervalues(self.paths):
            x.resolve(self.__dict__,keepUnresolvedSequencePlaceholders)
        for x in six.itervalues(self.endpaths):
            x.resolve(self.__dict__,keepUnresolvedSequencePlaceholders)
        if not self.schedule_() == None:
            for task in self.schedule_()._tasks:
                task.resolve(self.__dict__,keepUnresolvedSequencePlaceholders)

    def prune(self,verbose=False,keepUnresolvedSequencePlaceholders=False):
        """ Remove clutter from the process that we think is unnecessary:
        tracked PSets, VPSets and unused modules and sequences. If a Schedule has been set, then Paths and EndPaths
        not in the schedule will also be removed, along with an modules and sequences used only by
        those removed Paths and EndPaths. The keepUnresolvedSequencePlaceholders keeps also unresolved TaskPlaceholders."""
# need to update this to only prune psets not on refToPSets
# but for now, remove the delattr
#        for name in self.psets_():
#            if getattr(self,name).isTracked():
#                delattr(self, name)
        for name in self.vpsets_():
            delattr(self, name)
        #first we need to resolve any SequencePlaceholders being used
        self.resolve(keepUnresolvedSequencePlaceholders)
        usedModules = set()
        unneededPaths = set()
        tasks = list()
        tv = TaskVisitor(tasks)
        if self.schedule_():
            usedModules=set(self.schedule_().moduleNames())
            #get rid of unused paths
            schedNames = set(( x.label_() for x in self.schedule_()))
            names = set(self.paths)
            names.update(set(self.endpaths))
            unneededPaths = names - schedNames
            for n in unneededPaths:
                delattr(self,n)
            for t in self.schedule_().tasks():
                tv.enter(t)
                t.visit(tv)
                tv.leave(t)
        else:
            pths = list(six.itervalues(self.paths))
            pths.extend(six.itervalues(self.endpaths))
            temp = Schedule(*pths)
            usedModules=set(temp.moduleNames())
        unneededModules = self._pruneModules(self.producers_(), usedModules)
        unneededModules.update(self._pruneModules(self.switchProducers_(), usedModules))
        unneededModules.update(self._pruneModules(self.filters_(), usedModules))
        unneededModules.update(self._pruneModules(self.analyzers_(), usedModules))
        #remove sequences and tasks that do not appear in remaining paths and endpaths
        seqs = list()
        sv = SequenceVisitor(seqs)
        for p in six.itervalues(self.paths):
            p.visit(sv)
            p.visit(tv)
        for p in six.itervalues(self.endpaths):
            p.visit(sv)
            p.visit(tv)
        def removeUnneeded(seqOrTasks, allSequencesOrTasks):
            _keepSet = set(( s for s in seqOrTasks if s.hasLabel_()))
            _availableSet = set(six.itervalues(allSequencesOrTasks))
            _unneededSet = _availableSet-_keepSet
            _unneededLabels = []
            for s in _unneededSet:
                _unneededLabels.append(s.label_())
                delattr(self,s.label_())
            return _unneededLabels
        unneededSeqLabels = removeUnneeded(seqs, self.sequences)
        unneededTaskLabels = removeUnneeded(tasks, self.tasks)
        if verbose:
            print("prune removed the following:")
            print("  modules:"+",".join(unneededModules))
            print("  tasks:"+",".join(unneededTaskLabels))
            print("  sequences:"+",".join(unneededSeqLabels))
            print("  paths/endpaths:"+",".join(unneededPaths))
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
        self._insertOneInto(adaptor,  "@all_sources", self.source_(), True)
        self._insertOneInto(adaptor,  "@all_loopers", self.looper_(), True)
        self._insertSubProcessesInto(adaptor, "@all_subprocesses", self.subProcesses_(), False)
        self._insertManyInto(adaptor, "@all_esprefers", self.es_prefers_(), True)
        self._insertManyInto(adaptor, "@all_aliases", self.aliases_(), True)
        # This will visit all the paths and endpaths that are scheduled to run,
        # as well as the Tasks associated to them and the schedule. It remembers
        # the modules, ESSources, ESProducers, and services it visits.
        nodeVisitor = NodeVisitor()
        self._insertPaths(adaptor, nodeVisitor)
        all_modules_onTasksOrScheduled = { key:value for key, value in six.iteritems(all_modules) if value in nodeVisitor.modules }
        self._insertManyInto(adaptor, "@all_modules", all_modules_onTasksOrScheduled, True)
        all_switches = self.switchProducers_().copy()
        all_switches_onTasksOrScheduled = {key:value for key, value in six.iteritems(all_switches) if value in nodeVisitor.modules }
        self._insertSwitchProducersInto(adaptor, "@all_modules", "@all_aliases", all_switches_onTasksOrScheduled, True)
        # Same as nodeVisitor except this one visits all the Tasks attached
        # to the process.
        processNodeVisitor = NodeVisitor()
        for pTask in six.itervalues(self.tasks):
            pTask.visit(processNodeVisitor)
        esProducersToEnable = {}
        for esProducerName, esProducer in six.iteritems(self.es_producers_()):
            if esProducer in nodeVisitor.esProducers or not (esProducer in processNodeVisitor.esProducers):
                esProducersToEnable[esProducerName] = esProducer
        self._insertManyInto(adaptor, "@all_esmodules", esProducersToEnable, True)
        esSourcesToEnable = {}
        for esSourceName, esSource in six.iteritems(self.es_sources_()):
            if esSource in nodeVisitor.esSources or not (esSource in processNodeVisitor.esSources):
                esSourcesToEnable[esSourceName] = esSource
        self._insertManyInto(adaptor, "@all_essources", esSourcesToEnable, True)
        #handle services differently
        services = []
        for serviceName, serviceObject in six.iteritems(self.services_()):
            if serviceObject in nodeVisitor.services or not (serviceObject in processNodeVisitor.services):
                serviceObject.insertInto(ServiceInjectorAdaptor(adaptor,services))
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
           name of the C++ types in the Record that are being preferred, e.g.,
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
            for name, value in six.iteritems(d):
                if value.type_() == esname:
                    if found:
                        raise RuntimeError("More than one ES module for "+esname)
                    found = True
                    self.__setattr__(esname+"_prefer",  ESPrefer(d[esname].type_()) )
            return found


class ProcessFragment(object):
    def __init__(self, process):
        if isinstance(process, Process):
            self.__process = process
        elif isinstance(process, str):
            self.__process = Process(process)
            #make sure we do not override the defaults
            del self.__process.options
            del self.__process.maxEvents
            del self.__process.maxLuminosityBlocks
        else:
            raise TypeError('a ProcessFragment can only be constructed from an existig Process or from process name')
    def __dir__(self):
        return [ x for x in dir(self.__process) if isinstance(getattr(self.__process, x), _ConfigureComponent) ]
    def __getattribute__(self, name):
        if name == '_ProcessFragment__process':
            return object.__getattribute__(self, '_ProcessFragment__process')
        else:
            return getattr(self.__process, name)
    def __setattr__(self, name, value):
        if name == '_ProcessFragment__process':
            object.__setattr__(self, name, value)
        else:
            setattr(self.__process, name, value)
    def __delattr__(self, name):
        if name == '_ProcessFragment__process':
            pass
        else:
            return delattr(self.__process, name)


class FilteredStream(dict):
    """a dictionary with fixed keys"""
    def _blocked_attribute(obj):
        raise AttributeError("An FilteredStream defintion cannot be modified after creation.")
    _blocked_attribute = property(_blocked_attribute)
    __setattr__ = __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute
    def __new__(cls, *args, **kw):
        new = dict.__new__(cls)
        dict.__init__(new, *args, **kw)
        keys = sorted(kw.keys())
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

class SubProcess(_Unlabelable):
    """Allows embedding another process within a parent process. This allows one to 
    chain processes together directly in one cmsRun job rather than having to run
    separate jobs that are connected via a temporary file.
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
    def dumpPython(self, options=PrintOptions()):
        out = "parentProcess"+str(hash(self))+" = process\n"
        out += self.__process.dumpPython()
        out += "childProcess = process\n"
        out += "process = parentProcess"+str(hash(self))+"\n"
        out += "process.addSubProcess(cms.SubProcess(process = childProcess, SelectEvents = "+self.__SelectEvents.dumpPython(options) +", outputCommands = "+self.__outputCommands.dumpPython(options) +"))"
        return out
    def getProcessName(self):
        return self.__process.name_()
    def process(self):
        return self.__process
    def SelectEvents(self):
        return self.__SelectEvents
    def outputCommands(self):
        return self.__outputCommands
    def type_(self):
        return 'subProcess'
    def nameInProcessDesc_(self,label):
        return label
    def _place(self,label,process):
        process._placeSubProcess('subProcess',self)
    def getSubProcessPSet(self,parameterSet):
        topPSet = parameterSet.newPSet()
        self.__process.fillProcessDesc(topPSet)
        subProcessPSet = parameterSet.newPSet()
        self.__SelectEvents.insertInto(subProcessPSet,"SelectEvents")
        self.__outputCommands.insertInto(subProcessPSet,"outputCommands")
        subProcessPSet.addPSet(False,"process",topPSet)
        return subProcessPSet

class _ParameterModifier(object):
    """Helper class for Modifier that takes key/value pairs and uses them to reset parameters of the object"""
    def __init__(self,args):
        self.__args = args
    def __call__(self,obj):
        params = {}
        for k in six.iterkeys(self.__args):
            if hasattr(obj,k):
                params[k] = getattr(obj,k)
        _modifyParametersFromDict(params, self.__args, self._raiseUnknownKey)
        for k in six.iterkeys(self.__args):
            if k in params:
                setattr(obj,k,params[k])
            else:
                #the parameter must have been removed
                delattr(obj,k)
    @staticmethod
    def _raiseUnknownKey(key):
        raise KeyError("Unknown parameter name "+key+" specified while calling Modifier")

class _BoolModifierBase(object):
    """A helper base class for _AndModifier, _InvertModifier, and _OrModifier to contain the common code"""
    def __init__(self, lhs, rhs=None):
        self._lhs = lhs
        if rhs is not None:
            self._rhs = rhs
    def toModify(self,obj, func=None,**kw):
        Modifier._toModifyCheck(obj,func,**kw)
        if not self._isChosen():
            return
        Modifier._toModify(obj,func,**kw)
    def toReplaceWith(self,toObj,fromObj):
        Modifier._toReplaceWithCheck(toObj,fromObj)
        if not self._isChosen():
            return
        Modifier._toReplaceWith(toObj,fromObj)
    def makeProcessModifier(self,func):
        """This is used to create a ProcessModifer that can perform actions on the process as a whole.
            This takes as argument a callable object (e.g. function) that takes as its sole argument an instance of Process.
            In order to work, the value returned from this function must be assigned to a uniquely named variable."""
        return ProcessModifier(self,func)
    def __and__(self, other):
        return _AndModifier(self,other)
    def __invert__(self):
        return _InvertModifier(self)
    def __or__(self, other):
        return _OrModifier(self,other)

class _AndModifier(_BoolModifierBase):
    """A modifier which only applies if multiple Modifiers are chosen"""
    def __init__(self, lhs, rhs):
        super(_AndModifier,self).__init__(lhs, rhs)
    def _isChosen(self):
        return self._lhs._isChosen() and self._rhs._isChosen()

class _InvertModifier(_BoolModifierBase):
    """A modifier which only applies if a Modifier is not chosen"""
    def __init__(self, lhs):
        super(_InvertModifier,self).__init__(lhs)
    def _isChosen(self):
        return not self._lhs._isChosen()

class _OrModifier(_BoolModifierBase):
    """A modifier which only applies if at least one of multiple Modifiers is chosen"""
    def __init__(self, lhs, rhs):
        super(_OrModifier,self).__init__(lhs, rhs)
    def _isChosen(self):
        return self._lhs._isChosen() or self._rhs._isChosen()


class Modifier(object):
    """This class is used to define standard modifications to a Process.
    An instance of this class is declared to denote a specific modification,e.g. era2017 could
    reconfigure items in a process to match our expectation of running in 2017. Once declared,
    these Modifier instances are imported into a configuration and items that need to be modified
    are then associated with the Modifier and with the action to do the modification.
    The registered modifications will only occur if the Modifier was passed to 
    the cms.Process' constructor.
    """
    def __init__(self):
        self.__processModifiers = []
        self.__chosen = False
    def makeProcessModifier(self,func):
        """This is used to create a ProcessModifer that can perform actions on the process as a whole.
           This takes as argument a callable object (e.g. function) that takes as its sole argument an instance of Process.
           In order to work, the value returned from this function must be assigned to a uniquely named variable.
        """
        return ProcessModifier(self,func)
    @staticmethod
    def _toModifyCheck(obj,func,**kw):
        if func is not None and len(kw) != 0:
            raise TypeError("toModify takes either two arguments or one argument and key/value pairs")
    def toModify(self,obj, func=None,**kw):
        """This is used to register an action to be performed on the specific object. Two different forms are allowed
        Form 1: A callable object (e.g. function) can be passed as the second. This callable object is expected to take one argument
        that will be the object passed in as the first argument.
        Form 2: A list of parameter name, value pairs can be passed
           mod.toModify(foo, fred=cms.int32(7), barney = cms.double(3.14))
        This form can also be used to remove a parameter by passing the value of None
            #remove the parameter foo.fred       
            mod.toModify(foo, fred = None)
        Additionally, parameters embedded within PSets can also be modified using a dictionary
            #change foo.fred.pebbles to 3 and foo.fred.friend to "barney"
            mod.toModify(foo, fred = dict(pebbles = 3, friend = "barney)) )
        """
        Modifier._toModifyCheck(obj,func,**kw)
        if not self._isChosen():
            return
        Modifier._toModify(obj,func,**kw)
    @staticmethod
    def _toModify(obj,func,**kw):
        if func is not None:
            func(obj)
        else:
            temp =_ParameterModifier(kw)
            temp(obj)
    @staticmethod
    def _toReplaceWithCheck(toObj,fromObj):
        if not isinstance(fromObj, type(toObj)):
            raise TypeError("toReplaceWith requires both arguments to be the same class type")
    def toReplaceWith(self,toObj,fromObj):
        """If the Modifier is chosen the internals of toObj will be associated with the internals of fromObj
        """
        Modifier._toReplaceWithCheck(toObj,fromObj)
        if not self._isChosen():
            return
        Modifier._toReplaceWith(toObj,fromObj)
    @staticmethod
    def _toReplaceWith(toObj,fromObj):
        if isinstance(fromObj,_ModuleSequenceType):
            toObj._seq = fromObj._seq
            toObj._tasks = fromObj._tasks
        elif isinstance(fromObj,Task):
            toObj._collection = fromObj._collection
        elif isinstance(fromObj,_Parameterizable):
            #clear old items just incase fromObj is not a complete superset of toObj
            for p in toObj.parameterNames_():
                delattr(toObj,p)
            for p in fromObj.parameterNames_():
                setattr(toObj,p,getattr(fromObj,p))
            if isinstance(fromObj,_TypedParameterizable):
                toObj._TypedParameterizable__type = fromObj._TypedParameterizable__type

        else:
            raise TypeError("toReplaceWith does not work with type "+str(type(toObj)))

    def _setChosen(self):
        """Should only be called by cms.Process instances"""
        self.__chosen = True
    def _isChosen(self):
        return self.__chosen
    def __and__(self, other):
        return _AndModifier(self,other)
    def __invert__(self):
        return _InvertModifier(self)
    def __or__(self, other):
        return _OrModifier(self,other)
    def _isOrContains(self, other):
        return self == other


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
    def _isChosen(self):
        return self.__chosen
    def copyAndExclude(self, toExclude):
        """Creates a new ModifierChain which is a copy of
          this ModifierChain but excludes any Modifier or
          ModifierChain in the list toExclude.
          The exclusion is done recursively down the chain.
          """
        newMods = []
        for m in self.__chain:
            if m not in toExclude:
                s = m
                if isinstance(m,ModifierChain):
                    s = m.__copyIfExclude(toExclude)
                newMods.append(s)
        return ModifierChain(*newMods)
    def __copyIfExclude(self,toExclude):
        shouldCopy = False
        for m in toExclude:
            if self._isOrContains(m):
                shouldCopy = True
                break
        if shouldCopy:
            return self.copyAndExclude(toExclude)
        return self
    def _isOrContains(self, other):
        if self is other:
            return True
        for m in self.__chain:
            if m._isOrContains(other):
                return True
        return False

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
        if self.__modifier._isChosen():
            if process not in self.__seenProcesses:
                self.__func(process)
                self.__seenProcesses.add(process)

if __name__=="__main__":
    import unittest
    import copy

    def _lineDiff(newString, oldString):
        newString = ( x for x in newString.split('\n') if len(x) > 0)
        oldString = [ x for x in oldString.split('\n') if len(x) > 0]
        diff = []
        oldStringLine = 0
        for l in newString:
            if oldStringLine >= len(oldString):
                diff.append(l)
                continue
            if l == oldString[oldStringLine]:
                oldStringLine +=1
                continue
            diff.append(l)
        return "\n".join( diff )

    class TestMakePSet(object):
        """Has same interface as the C++ object that creates PSets
        """
        def __init__(self):
            self.values = dict()
        def __insertValue(self,tracked,label,value):
            self.values[label]=(tracked,value)
        def __getValue(self,tracked,label):
            pair = self.values[label]
            if pair[0] != tracked:
               raise Exception("Asked for %s parameter '%s', but it is %s" % ("tracked" if tracked else "untracked",
                                                                              label,
                                                                              "tracked" if pair[0] else "untracked"))
            return pair[1]
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
        def getVString(self,tracked,label):
            return self.__getValue(tracked, label)
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

    class SwitchProducerTest(SwitchProducer):
        def __init__(self, **kargs):
            super(SwitchProducerTest,self).__init__(
                dict(
                    test1 = lambda: (True, -10),
                    test2 = lambda: (True, -9),
                    test3 = lambda: (True, -8),
                    test4 = lambda: (True, -7)
                ), **kargs)
    specialImportRegistry.registerSpecialImportForType(SwitchProducerTest, "from test import SwitchProducerTest")

    class TestModuleCommand(unittest.TestCase):
        def setUp(self):
            """Nothing to do """
            None
        def testParameterizable(self):
            p = _Parameterizable()
            self.assertEqual(len(p.parameterNames_()),0)
            p.a = int32(1)
            self.assertTrue('a' in p.parameterNames_())
            self.assertEqual(p.a.value(), 1)
            p.a = 10
            self.assertEqual(p.a.value(), 10)
            p.a = untracked(int32(1))
            self.assertEqual(p.a.value(), 1)
            self.assertFalse(p.a.isTracked())
            p.a = untracked.int32(1)
            self.assertEqual(p.a.value(), 1)
            self.assertFalse(p.a.isTracked())
            p = _Parameterizable(foo=int32(10), bar = untracked(double(1.0)))
            self.assertEqual(p.foo.value(), 10)
            self.assertEqual(p.bar.value(),1.0)
            self.assertFalse(p.bar.isTracked())
            self.assertRaises(TypeError,setattr,(p,'c',1))
            p = _Parameterizable(a=PSet(foo=int32(10), bar = untracked(double(1.0))))
            self.assertEqual(p.a.foo.value(),10)
            self.assertEqual(p.a.bar.value(),1.0)
            p.b = untracked(PSet(fii = int32(1)))
            self.assertEqual(p.b.fii.value(),1)
            self.assertFalse(p.b.isTracked())
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
            self.assertTrue( 'a' in p.analyzers_() )
            self.assertTrue( 'a' in p.analyzers)
            p.add_(Service("MessageLogger"))
            self.assertTrue('MessageLogger' in p.services_())
            self.assertEqual(p.MessageLogger.type_(), "MessageLogger")
            p.Tracer = Service("Tracer")
            self.assertTrue('Tracer' in p.services_())
            self.assertRaises(TypeError, setattr, *(p,'b',"this should fail"))
            self.assertRaises(TypeError, setattr, *(p,'bad',Service("MessageLogger")))
            self.assertRaises(ValueError, setattr, *(p,'bad',Source("PoolSource")))
            p.out = OutputModule("Outer")
            self.assertEqual(p.out.type_(), 'Outer')
            self.assertTrue( 'out' in p.outputModules_() )

            p.geom = ESSource("GeomProd")
            self.assertTrue('geom' in p.es_sources_())
            p.add_(ESSource("ConfigDB"))
            self.assertTrue('ConfigDB' in p.es_sources_())

            p.aliasfoo1 = EDAlias(foo1 = VPSet(PSet(type = string("Foo1"))))
            self.assertTrue('aliasfoo1' in p.aliases_())

        def testProcessExtend(self):
            class FromArg(object):
                def __init__(self,*arg,**args):
                    for name in six.iterkeys(args):
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


            p = Process('test')
            p.a = EDProducer("MyProducer")
            p.t = Task(p.a)
            p.p = Path(p.t)
            self.assertRaises(ValueError, p.extend, FromArg(a = EDProducer("YourProducer")))
            self.assertRaises(ValueError, p.extend, FromArg(a = EDAlias()))
            self.assertRaises(ValueError, p.__setattr__, "a", EDAlias())

            p = Process('test')
            p.a = EDProducer("MyProducer")
            p.s = Sequence(p.a)
            p.p = Path(p.s)
            self.assertRaises(ValueError, p.extend, FromArg(a = EDProducer("YourProducer")))
            self.assertRaises(ValueError, p.extend, FromArg(a = EDAlias()))
            self.assertRaises(ValueError, p.__setattr__, "a", EDAlias())

        def testProcessDumpPython(self):
            self.assertEqual(Process("test").dumpPython(),
"""import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.maxEvents = cms.untracked.PSet(
    input = cms.optional.untracked.int32,
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

process.maxLuminosityBlocks = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

""")
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.p = Path(p.a)
            p.s = Sequence(p.a)
            p.r = Sequence(p.s)
            p.p2 = Path(p.s)
            p.schedule = Schedule(p.p2,p.p)
            d=p.dumpPython()
            self.assertEqual(_lineDiff(d,Process("test").dumpPython()),
"""process.a = cms.EDAnalyzer("MyAnalyzer")
process.s = cms.Sequence(process.a)
process.r = cms.Sequence(process.s)
process.p = cms.Path(process.a)
process.p2 = cms.Path(process.s)
process.schedule = cms.Schedule(*[ process.p2, process.p ])""")
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
            self.assertEqual(_lineDiff(d,Process("test").dumpPython()),
"""process.a = cms.EDAnalyzer("MyAnalyzer")
process.b = cms.EDAnalyzer("YourAnalyzer")
process.r = cms.Sequence(process.a)
process.s = cms.Sequence(process.r)
process.p = cms.Path(process.a)
process.p2 = cms.Path(process.r)
process.schedule = cms.Schedule(*[ process.p2, process.p ])""")
        #use an anonymous sequence
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.p = Path(p.a)
            s = Sequence(p.a)
            p.r = Sequence(s)
            p.p2 = Path(p.r)
            p.schedule = Schedule(p.p2,p.p)
            d=p.dumpPython()
            self.assertEqual(_lineDiff(d,Process("test").dumpPython()),
"""process.a = cms.EDAnalyzer("MyAnalyzer")
process.r = cms.Sequence((process.a))
process.p = cms.Path(process.a)
process.p2 = cms.Path(process.r)
process.schedule = cms.Schedule(*[ process.p2, process.p ])""")

            # include some tasks
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDProducer("bProducer")
            p.c = EDProducer("cProducer")
            p.d = EDProducer("dProducer")
            p.e = EDProducer("eProducer")
            p.f = EDProducer("fProducer")
            p.g = EDProducer("gProducer")
            p.task5 = Task()
            p.task3 = Task()
            p.task2 = Task(p.c, p.task3)
            p.task4 = Task(p.f, p.task2)
            p.task1 = Task(p.task5)
            p.task3.add(p.task1)
            p.p = Path(p.a)
            s = Sequence(p.a)
            p.r = Sequence(s)
            p.p2 = Path(p.r, p.task1, p.task2)
            p.schedule = Schedule(p.p2,p.p,tasks=[p.task3,p.task4, p.task5])
            d=p.dumpPython()
            self.assertEqual(_lineDiff(d,Process("test").dumpPython()),
"""process.b = cms.EDProducer("bProducer")
process.c = cms.EDProducer("cProducer")
process.d = cms.EDProducer("dProducer")
process.e = cms.EDProducer("eProducer")
process.f = cms.EDProducer("fProducer")
process.g = cms.EDProducer("gProducer")
process.a = cms.EDAnalyzer("MyAnalyzer")
process.task5 = cms.Task()
process.task1 = cms.Task(process.task5)
process.task3 = cms.Task(process.task1)
process.task2 = cms.Task(process.c, process.task3)
process.task4 = cms.Task(process.f, process.task2)
process.r = cms.Sequence((process.a))
process.p = cms.Path(process.a)
process.p2 = cms.Path(process.r, process.task1, process.task2)
process.schedule = cms.Schedule(*[ process.p2, process.p ], tasks=[process.task3, process.task4, process.task5])""")
            # only tasks
            p = Process("test")
            p.d = EDProducer("dProducer")
            p.e = EDProducer("eProducer")
            p.f = EDProducer("fProducer")
            p.g = EDProducer("gProducer")
            p.task1 = Task(p.d, p.e)
            task2 = Task(p.f, p.g)
            p.schedule = Schedule(tasks=[p.task1,task2])
            d=p.dumpPython()
            self.assertEqual(_lineDiff(d,Process("test").dumpPython()),
"""process.d = cms.EDProducer("dProducer")
process.e = cms.EDProducer("eProducer")
process.f = cms.EDProducer("fProducer")
process.g = cms.EDProducer("gProducer")
process.task1 = cms.Task(process.d, process.e)
process.schedule = cms.Schedule(tasks=[cms.Task(process.f, process.g), process.task1])""")
            # empty schedule
            p = Process("test")
            p.schedule = Schedule()
            d=p.dumpPython()
            self.assertEqual(_lineDiff(d,Process('test').dumpPython()),
"""process.schedule = cms.Schedule()""")

            s = Sequence()
            a = EDProducer("A")
            s2 = Sequence(a)
            s2 += s
            process = Process("DUMP")
            process.a = a
            process.s2 = s2
            d=process.dumpPython()
            self.assertEqual(_lineDiff(d,Process('DUMP').dumpPython()),
"""process.a = cms.EDProducer("A")
process.s2 = cms.Sequence(process.a)""")
            s = Sequence()
            s1 = Sequence(s)
            a = EDProducer("A")
            s3 = Sequence(a+a)
            s2 = Sequence(a+s3)
            s2 += s1
            process = Process("DUMP")
            process.a = a
            process.s2 = s2
            d=process.dumpPython()
            self.assertEqual(_lineDiff(d,Process('DUMP').dumpPython()),
"""process.a = cms.EDProducer("A")
process.s2 = cms.Sequence(process.a+(process.a+process.a))""")

        def testSecSource(self):
            p = Process('test')
            p.a = SecSource("MySecSource")
            self.assertEqual(_lineDiff(p.dumpPython(),Process('test').dumpPython()),'process.a = cms.SecSource("MySecSource")')

        def testGlobalReplace(self):
            p = Process('test')
            p.a = EDAnalyzer("MyAnalyzer")
            old = p.a
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.d = EDProducer("MyProducer")
            old2 = p.d
            p.t1 = Task(p.d)
            t2 = Task(p.d)
            t3 = Task(p.d)
            t4 = Task(p.d)
            t5 = Task(p.d)
            t6 = Task(p.d)
            s = Sequence(p.a*p.b)
            p.s4 = Sequence(p.a*p.b)
            s.associate(t2)
            p.s4.associate(t2)
            p.p = Path(p.c+s+p.a)
            p.p2 = Path(p.c+p.s4+p.a)
            p.e3 = EndPath(p.c+s+p.a)
            new = EDAnalyzer("NewAnalyzer")
            new2 = EDProducer("NewProducer")
            visitor1 = NodeVisitor()
            p.p.visit(visitor1)
            self.assertTrue(visitor1.modules == set([old,old2,p.b,p.c]))
            p.schedule = Schedule(tasks=[t6])
            p.globalReplace("a",new)
            p.globalReplace("d",new2)
            visitor2 = NodeVisitor()
            p.p.visit(visitor2)
            self.assertTrue(visitor2.modules == set([new,new2,p.b,p.c]))
            self.assertEqual(p.p.dumpPython()[:-1], "cms.Path(process.c+process.a+process.b+process.a, cms.Task(process.d))")
            visitor_p2 = NodeVisitor()
            p.p2.visit(visitor_p2)
            self.assertTrue(visitor_p2.modules == set([new,new2,p.b,p.c]))
            self.assertEqual(p.p2.dumpPython()[:-1], "cms.Path(process.c+process.s4+process.a)")
            visitor3 = NodeVisitor()
            p.e3.visit(visitor3)
            self.assertTrue(visitor3.modules == set([new,new2,p.b,p.c]))
            visitor4 = NodeVisitor()
            p.s4.visit(visitor4)
            self.assertTrue(visitor4.modules == set([new,new2,p.b]))
            self.assertEqual(p.s4.dumpPython()[:-1],"cms.Sequence(process.a+process.b, cms.Task(process.d))")
            visitor5 = NodeVisitor()
            p.t1.visit(visitor5)
            self.assertTrue(visitor5.modules == set([new2]))
            visitor6 = NodeVisitor()
            listOfTasks = list(p.schedule._tasks)
            listOfTasks[0].visit(visitor6)
            self.assertTrue(visitor6.modules == set([new2]))

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

        def testServiceInProcess(self):
            service = Service("d")
            self.assertFalse(service._inProcess)
            process = Process("test")
            process.d = service
            self.assertTrue(service._inProcess)
            service2 = Service("d")
            process.d = service2
            self.assertFalse(service._inProcess)
            self.assertTrue(service2._inProcess)
            del process.d
            self.assertFalse(service2._inProcess)

        def testTask(self):

            # create some objects to use in tests
            edanalyzer = EDAnalyzer("a")
            edproducer = EDProducer("b")
            edproducer2 = EDProducer("b2")
            edproducer3 = EDProducer("b3")
            edproducer4 = EDProducer("b4")
            edproducer8 = EDProducer("b8")
            edproducer9 = EDProducer("b9")
            edfilter = EDFilter("c")
            service = Service("d")
            service3 = Service("d")
            essource = ESSource("e")
            esproducer = ESProducer("f")
            testTask2 = Task()

            # test adding things to Tasks
            testTask1 = Task(edproducer, edfilter)
            self.assertRaises(RuntimeError, testTask1.add, edanalyzer)
            testTask1.add(essource, service)
            testTask1.add(essource, esproducer)
            testTask1.add(testTask2)
            coll = testTask1._collection
            self.assertTrue(edproducer in coll)
            self.assertTrue(edfilter in coll)
            self.assertTrue(service in coll)
            self.assertTrue(essource in coll)
            self.assertTrue(esproducer in coll)
            self.assertTrue(testTask2 in coll)
            self.assertTrue(len(coll) == 6)
            self.assertTrue(len(testTask2._collection) == 0)

            taskContents = []
            for i in testTask1:
                taskContents.append(i)
            self.assertTrue(taskContents == [edproducer, edfilter, essource, service, esproducer, testTask2])

            # test attaching Task to Process
            process = Process("test")

            process.mproducer = edproducer
            process.mproducer2 = edproducer2
            process.mfilter = edfilter
            process.messource = essource
            process.mesproducer = esproducer
            process.d = service

            testTask3 = Task(edproducer, edproducer2)
            testTask1.add(testTask3)
            process.myTask1 = testTask1

            # test the validation that occurs when attaching a Task to a Process
            # first a case that passes, then one the fails on an EDProducer
            # then one that fails on a service
            l = set()
            visitor = NodeNameVisitor(l)
            testTask1.visit(visitor)
            self.assertTrue(l == set(['mesproducer', 'mproducer', 'mproducer2', 'mfilter', 'd', 'messource']))
            l2 = testTask1.moduleNames
            self.assertTrue(l == set(['mesproducer', 'mproducer', 'mproducer2', 'mfilter', 'd', 'messource']))

            testTask4 = Task(edproducer3)
            l.clear()
            self.assertRaises(RuntimeError, testTask4.visit, visitor)
            try:
                process.myTask4 = testTask4
                self.assertTrue(False)
            except RuntimeError:
                pass

            testTask5 = Task(service3)
            l.clear()
            self.assertRaises(RuntimeError, testTask5.visit, visitor)
            try:
                process.myTask5 = testTask5
                self.assertTrue(False)
            except RuntimeError:
                pass

            process.d = service3
            process.myTask5 = testTask5

            # test placement into the Process and the tasks property
            expectedDict = { 'myTask1' : testTask1, 'myTask5' : testTask5 }
            expectedFixedDict = DictTypes.FixedKeysDict(expectedDict);
            self.assertTrue(process.tasks == expectedFixedDict)
            self.assertTrue(process.tasks['myTask1'] == testTask1)
            self.assertTrue(process.myTask1 == testTask1)

            # test replacing an EDProducer in a Task when calling __settattr__
            # for the EDProducer on the Process.
            process.mproducer2 = edproducer4
            process.d = service
            l = list()
            visitor1 = ModuleNodeVisitor(l)
            testTask1.visit(visitor1)
            l.sort(key=lambda mod: mod.__str__())
            expectedList = sorted([edproducer,essource,esproducer,service,edfilter,edproducer,edproducer4],key=lambda mod: mod.__str__())
            self.assertTrue(expectedList == l)
            process.myTask6 = Task()
            process.myTask7 = Task()
            process.mproducer8 = edproducer8
            process.myTask8 = Task(process.mproducer8)
            process.myTask6.add(process.myTask7)
            process.myTask7.add(process.myTask8)
            process.myTask1.add(process.myTask6)
            process.myTask8.add(process.myTask5)

            testDict = process._itemsInDependencyOrder(process.tasks)
            expectedLabels = ["myTask5", "myTask8", "myTask7", "myTask6", "myTask1"]
            expectedTasks = [process.myTask5, process.myTask8, process.myTask7, process.myTask6, process.myTask1]
            index = 0
            for testLabel, testTask in testDict.items():
                self.assertTrue(testLabel == expectedLabels[index])
                self.assertTrue(testTask == expectedTasks[index])
                index += 1

            pythonDump = testTask1.dumpPython(PrintOptions())


            expectedPythonDump = 'cms.Task(process.d, process.mesproducer, process.messource, process.mfilter, process.mproducer, process.mproducer2, process.myTask6)\n'
            self.assertTrue(pythonDump == expectedPythonDump)

            process.myTask5 = Task()
            process.myTask100 = Task()
            process.mproducer9 = edproducer9
            sequence1 = Sequence(process.mproducer8, process.myTask1, process.myTask5, testTask2, testTask3)
            sequence2 = Sequence(process.mproducer8 + process.mproducer9)
            process.sequence3 = Sequence((process.mproducer8 + process.mfilter))
            sequence4 = Sequence()
            process.path1 = Path(process.mproducer+process.mproducer8+sequence1+sequence2+process.sequence3+sequence4)
            process.path1.associate(process.myTask1, process.myTask5, testTask2, testTask3)
            process.path11 = Path(process.mproducer+process.mproducer8+sequence1+sequence2+process.sequence3+ sequence4,process.myTask1, process.myTask5, testTask2, testTask3, process.myTask100)
            process.path2 = Path(process.mproducer)
            process.path3 = Path(process.mproducer9+process.mproducer8,testTask2)

            self.assertTrue(process.path1.dumpPython(PrintOptions()) == 'cms.Path(process.mproducer+process.mproducer8+cms.Sequence(process.mproducer8, cms.Task(), cms.Task(process.None, process.mproducer), process.myTask1, process.myTask5)+(process.mproducer8+process.mproducer9)+process.sequence3, cms.Task(), cms.Task(process.None, process.mproducer), process.myTask1, process.myTask5)\n')

            self.assertTrue(process.path11.dumpPython(PrintOptions()) == 'cms.Path(process.mproducer+process.mproducer8+cms.Sequence(process.mproducer8, cms.Task(), cms.Task(process.None, process.mproducer), process.myTask1, process.myTask5)+(process.mproducer8+process.mproducer9)+process.sequence3, cms.Task(), cms.Task(process.None, process.mproducer), process.myTask1, process.myTask100, process.myTask5)\n')

            # test NodeNameVisitor and moduleNames
            l = set()
            nameVisitor = NodeNameVisitor(l)
            process.path1.visit(nameVisitor)
            self.assertTrue(l == set(['mproducer', 'd', 'mesproducer', None, 'mproducer9', 'mproducer8', 'messource', 'mproducer2', 'mfilter']))
            self.assertTrue(process.path1.moduleNames() == set(['mproducer', 'd', 'mesproducer', None, 'mproducer9', 'mproducer8', 'messource', 'mproducer2', 'mfilter']))

            # test copy
            process.mproducer10 = EDProducer("b10")
            process.path21 = process.path11.copy()
            process.path21.replace(process.mproducer, process.mproducer10)

            self.assertTrue(process.path11.dumpPython(PrintOptions()) == 'cms.Path(process.mproducer+process.mproducer8+cms.Sequence(process.mproducer8, cms.Task(), cms.Task(process.None, process.mproducer), process.myTask1, process.myTask5)+(process.mproducer8+process.mproducer9)+process.sequence3, cms.Task(), cms.Task(process.None, process.mproducer), process.myTask1, process.myTask100, process.myTask5)\n')

            # Some peculiarities of the way things work show up here. dumpPython sorts tasks and
            # removes duplication at the level of strings. The Task and Sequence objects themselves
            # remove duplicate tasks in their contents if the instances are the same (exact same python
            # object id which is not the same as the string representation being the same).
            # Also note that the mutating visitor replaces sequences and tasks that have
            # modified contents with their modified contents, it does not modify the sequence
            # or task itself.
            self.assertTrue(process.path21.dumpPython(PrintOptions()) == 'cms.Path(process.mproducer10+process.mproducer8+process.mproducer8+(process.mproducer8+process.mproducer9)+process.sequence3, cms.Task(), cms.Task(process.None, process.mproducer10), cms.Task(process.d, process.mesproducer, process.messource, process.mfilter, process.mproducer10, process.mproducer2, process.myTask6), process.myTask100, process.myTask5)\n')

            process.path22 = process.path21.copyAndExclude([process.d, process.mesproducer, process.mfilter])
            self.assertTrue(process.path22.dumpPython(PrintOptions()) == 'cms.Path(process.mproducer10+process.mproducer8+process.mproducer8+(process.mproducer8+process.mproducer9)+process.mproducer8, cms.Task(), cms.Task(process.None, process.mproducer10), cms.Task(process.messource, process.mproducer10, process.mproducer2, process.myTask6), process.myTask100, process.myTask5)\n')

            process.path23 = process.path22.copyAndExclude([process.messource, process.mproducer10])
            self.assertTrue(process.path23.dumpPython(PrintOptions()) == 'cms.Path(process.mproducer8+process.mproducer8+(process.mproducer8+process.mproducer9)+process.mproducer8, cms.Task(), cms.Task(process.None), cms.Task(process.mproducer2, process.myTask6), process.myTask100, process.myTask5)\n')

            process.a = EDAnalyzer("MyAnalyzer")
            process.b = OutputModule("MyOutputModule")
            process.c = EDFilter("MyFilter")
            process.d = EDProducer("MyProducer")
            process.e = ESProducer("MyESProducer")
            process.f = ESSource("MyESSource")
            process.g = ESProducer("g")
            process.path24 = Path(process.a+process.b+process.c+process.d)
            process.path25 = process.path24.copyAndExclude([process.a,process.b,process.c])
            self.assertTrue(process.path25.dumpPython() == 'cms.Path(process.d)\n')
            #print process.path3
            #print process.dumpPython()

            process.path200 = EndPath(Sequence(process.c,Task(process.e)))
            process.path200.replace(process.c,process.b)
            process.path200.replace(process.e,process.f)
            self.assertEqual(process.path200.dumpPython(), "cms.EndPath(process.b, cms.Task(process.f))\n")
            process.path200.replace(process.b,process.c)
            process.path200.replace(process.f,process.e)
            self.assertEqual(process.path200.dumpPython(), "cms.EndPath(process.c, cms.Task(process.e))\n")
            process.path200.replace(process.c,process.a)
            process.path200.replace(process.e,process.g)
            self.assertEqual(process.path200.dumpPython(), "cms.EndPath(process.a, cms.Task(process.g))\n")
            process.path200.replace(process.a,process.c)
            process.path200.replace(process.g,process.e)
            self.assertEqual(process.path200.dumpPython(), "cms.EndPath(process.c, cms.Task(process.e))\n")


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
            pathx = Path(p.a*(p.b+ignore(p.c)))
            self.assertEqual(str(path),'a+b+~c')
            p.es = ESProducer("AnESProducer")
            self.assertRaises(TypeError,Path,p.es)

            t = Path()
            self.assertTrue(t.dumpPython(PrintOptions()) == 'cms.Path()\n')

            t = Path(p.a)
            self.assertTrue(t.dumpPython(PrintOptions()) == 'cms.Path(process.a)\n')

            t = Path(Task())
            self.assertTrue(t.dumpPython(PrintOptions()) == 'cms.Path(cms.Task())\n')

            t = Path(p.a, Task())
            self.assertTrue(t.dumpPython(PrintOptions()) == 'cms.Path(process.a, cms.Task())\n')

            p.prod = EDProducer("prodName")
            p.t1 = Task(p.prod)
            t = Path(p.a, p.t1, Task(), p.t1)
            self.assertTrue(t.dumpPython(PrintOptions()) == 'cms.Path(process.a, cms.Task(), process.t1)\n')

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

        def testContains(self):

            a = EDProducer("a")
            b = EDProducer("b")
            c = EDProducer("c")
            d = EDProducer("d")
            e = EDProducer("e")
            f = EDProducer("f")
            g = EDProducer("g")
            h = EDProducer("h")
            i = EDProducer("i")
            j = EDProducer("j")
            k = EDProducer("k")
            l = EDProducer("l")
            m = EDProducer("m")
            n = EDProducer("n")

            seq1 = Sequence(e)
            task1 = Task(g)
            path = Path(a * c * seq1, task1)

            self.assertTrue(path.contains(a))
            self.assertFalse(path.contains(b))
            self.assertTrue(path.contains(c))
            self.assertFalse(path.contains(d))
            self.assertTrue(path.contains(e))
            self.assertFalse(path.contains(f))
            self.assertTrue(path.contains(g))

            endpath = EndPath(h * i)
            self.assertFalse(endpath.contains(b))
            self.assertTrue(endpath.contains(i))

            seq = Sequence(a * c)
            self.assertFalse(seq.contains(b))
            self.assertTrue(seq.contains(c))

            task2 = Task(l)
            task = Task(j, k, task2)
            self.assertFalse(task.contains(b))
            self.assertTrue(task.contains(j))
            self.assertTrue(task.contains(k))
            self.assertTrue(task.contains(l))

            task3 = Task(m)
            path2 = Path(n)
            sch = Schedule(path, path2, tasks=[task,task3])
            self.assertFalse(sch.contains(b))
            self.assertTrue(sch.contains(a))
            self.assertTrue(sch.contains(c))
            self.assertTrue(sch.contains(e))
            self.assertTrue(sch.contains(g))
            self.assertTrue(sch.contains(n))
            self.assertTrue(sch.contains(j))
            self.assertTrue(sch.contains(k))
            self.assertTrue(sch.contains(l))
            self.assertTrue(sch.contains(m))

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
            self.assertTrue('b' in p.schedule.moduleNames())
            self.assertTrue(hasattr(p, 'b'))
            self.assertTrue(hasattr(p, 'c'))
            self.assertTrue(hasattr(p, 'd'))
            self.assertTrue(hasattr(p, 'path1'))
            self.assertTrue(hasattr(p, 'path2'))
            self.assertTrue(hasattr(p, 'path3'))
            p.prune()
            self.assertTrue('b' in p.schedule.moduleNames())
            self.assertTrue(hasattr(p, 'b'))
            self.assertTrue(not hasattr(p, 'c'))
            self.assertTrue(not hasattr(p, 'd'))
            self.assertTrue(hasattr(p, 'path1'))
            self.assertTrue(hasattr(p, 'path2'))
            self.assertTrue(not hasattr(p, 'path3'))

            self.assertTrue(len(p.schedule._tasks) == 0)

            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.d = EDAnalyzer("dAnalyzer")
            p.e = EDProducer("eProducer")
            p.f = EDProducer("fProducer")
            p.Tracer = Service("Tracer")
            p.path1 = Path(p.a)
            p.path2 = Path(p.b)
            p.path3 = Path(p.d)
            p.task1 = Task(p.e)
            p.task2 = Task(p.f, p.Tracer)
            s = Schedule(p.path1,p.path2,tasks=[p.task1,p.task2,p.task1])
            self.assertEqual(s[0],p.path1)
            self.assertEqual(s[1],p.path2)
            self.assertTrue(len(s._tasks) == 2)
            self.assertTrue(p.task1 in s._tasks)
            self.assertTrue(p.task2 in s._tasks)
            listOfTasks = list(s._tasks)
            self.assertTrue(len(listOfTasks) == 2)
            self.assertTrue(p.task1 == listOfTasks[0])
            self.assertTrue(p.task2 == listOfTasks[1])
            p.schedule = s
            self.assertTrue('b' in p.schedule.moduleNames())

            process2 = Process("test")
            process2.a = EDAnalyzer("MyAnalyzer")
            process2.e = EDProducer("eProducer")
            process2.path1 = Path(process2.a)
            process2.task1 = Task(process2.e)
            process2.schedule = Schedule(process2.path1,tasks=process2.task1)
            listOfTasks = list(process2.schedule._tasks)
            self.assertTrue(listOfTasks[0] == process2.task1)

            # test Schedule copy
            s2 = s.copy()
            self.assertEqual(s2[0],p.path1)
            self.assertEqual(s2[1],p.path2)
            self.assertTrue(len(s2._tasks) == 2)
            self.assertTrue(p.task1 in s2._tasks)
            self.assertTrue(p.task2 in s2._tasks)
            listOfTasks = list(s2._tasks)
            self.assertTrue(len(listOfTasks) == 2)
            self.assertTrue(p.task1 == listOfTasks[0])
            self.assertTrue(p.task2 == listOfTasks[1])

            names = s.moduleNames()
            self.assertTrue(names == set(['a', 'b', 'e', 'Tracer', 'f']))
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
            self.assertTrue('a' in s.moduleNames())
            self.assertTrue('b' in s.moduleNames())
            self.assertTrue('c' in s.moduleNames())
            p.path1 = path1
            p.schedule = s
            p.prune()
            self.assertTrue('a' in s.moduleNames())
            self.assertTrue('b' in s.moduleNames())
            self.assertTrue('c' in s.moduleNames())

        def testImplicitSchedule(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.path1 = Path(p.a)
            p.path2 = Path(p.b)
            self.assertTrue(p.schedule is None)
            pths = p.paths
            keys = pths.keys()
            self.assertEqual(pths[keys[0]],p.path1)
            self.assertEqual(pths[keys[1]],p.path2)
            p.prune()
            self.assertTrue(hasattr(p, 'a'))
            self.assertTrue(hasattr(p, 'b'))
            self.assertTrue(not hasattr(p, 'c'))
            self.assertTrue(hasattr(p, 'path1'))
            self.assertTrue(hasattr(p, 'path2'))


            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.path2 = Path(p.b)
            p.path1 = Path(p.a)
            self.assertTrue(p.schedule is None)
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
            self.assertTrue(not a.isModified())
            a.a1 = 1
            self.assertTrue(a.isModified())
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
        
        def testOptions(self):
            p = Process('test')
            self.assertEqual(p.options.numberOfThreads.value(),1)
            p.options.numberOfThreads = 8
            self.assertEqual(p.options.numberOfThreads.value(),8)
            p.options = PSet()
            self.assertEqual(p.options.numberOfThreads.value(),1)
            p.options = dict(numberOfStreams =2,
                             numberOfThreads =2)
            self.assertEqual(p.options.numberOfThreads.value(),2)
            self.assertEqual(p.options.numberOfStreams.value(),2)

        def testMaxEvents(self):
            p = Process("Test")
            p.maxEvents.input = 10
            self.assertEqual(p.maxEvents.input.value(),10)
            p = Process("Test")
            p.maxEvents.output = 10
            self.assertEqual(p.maxEvents.output.value(),10)
            p = Process("Test")
            p.maxEvents.output = PSet(out=untracked.int32(10))
            self.assertEqual(p.maxEvents.output.out.value(), 10)
            p = Process("Test")
            p.maxEvents = untracked.PSet(input = untracked.int32(5))
            self.assertEqual(p.maxEvents.input.value(), 5)

        
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
            self.assertEqual(_lineDiff(p.dumpPython(), Process('Test').dumpPython()),
"""process.juicer = cms.ESProducer("JuicerProducer")
process.ForceSource = cms.ESSource("ForceSource")
process.prefer("ForceSource")
process.prefer("juicer")""")
            p.prefer("juicer",fooRcd=vstring("Foo"))
            self.assertEqual(_lineDiff(p.dumpPython(), Process('Test').dumpPython()),
"""process.juicer = cms.ESProducer("JuicerProducer")
process.ForceSource = cms.ESSource("ForceSource")
process.prefer("ForceSource")
process.prefer("juicer",
    fooRcd = cms.vstring('Foo')
)""")

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
            process.addSubProcess(SubProcess(subProcess))
            d = process.dumpPython()
            equalD ="""parentProcess = process
process.a = cms.EDProducer("A")
process.Foo = cms.Service("Foo")
process.p = cms.Path(process.a)
childProcess = process
process = parentProcess
process.addSubProcess(cms.SubProcess(process = childProcess, SelectEvents = cms.untracked.PSet(
), outputCommands = cms.untracked.vstring()))"""
            equalD = equalD.replace("parentProcess","parentProcess"+str(hash(process.subProcesses_()[0])))
            self.assertEqual(_lineDiff(d,Process('Parent').dumpPython()+Process('Child').dumpPython()),equalD)
            p = TestMakePSet()
            process.fillProcessDesc(p)
            self.assertEqual((True,['a']),p.values["subProcesses"][1][0].values["process"][1].values['@all_modules'])
            self.assertEqual((True,['p']),p.values["subProcesses"][1][0].values["process"][1].values['@paths'])
            self.assertEqual({'@service_type':(True,'Foo')}, p.values["subProcesses"][1][0].values["process"][1].values["services"][1][0].values)
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
        def testSwitchProducer(self):
            proc = Process("test")
            proc.sp = SwitchProducerTest(test2 = EDProducer("Foo",
                                                            a = int32(1),
                                                            b = PSet(c = int32(2))),
                                         test1 = EDProducer("Bar",
                                                            aa = int32(11),
                                                            bb = PSet(cc = int32(12))))
            proc.a = EDProducer("A")
            proc.s = Sequence(proc.a + proc.sp)
            proc.t = Task(proc.a, proc.sp)
            proc.p = Path()
            proc.p.associate(proc.t)
            p = TestMakePSet()
            proc.fillProcessDesc(p)
            self.assertEqual((True,"EDProducer"), p.values["sp"][1].values["@module_edm_type"])
            self.assertEqual((True, "SwitchProducer"), p.values["sp"][1].values["@module_type"])
            self.assertEqual((True, "sp"), p.values["sp"][1].values["@module_label"])
            all_cases = copy.deepcopy(p.values["sp"][1].values["@all_cases"])
            all_cases[1].sort() # names of all cases come via dict, i.e. their order is undefined
            self.assertEqual((True, ["sp@test1", "sp@test2"]), all_cases)
            self.assertEqual((False, "sp@test2"), p.values["sp"][1].values["@chosen_case"])
            self.assertEqual(["a", "sp", "sp@test1", "sp@test2"], p.values["@all_modules"][1])
            self.assertEqual((True,"EDProducer"), p.values["sp@test1"][1].values["@module_edm_type"])
            self.assertEqual((True,"Bar"), p.values["sp@test1"][1].values["@module_type"])
            self.assertEqual((True,"EDProducer"), p.values["sp@test2"][1].values["@module_edm_type"])
            self.assertEqual((True,"Foo"), p.values["sp@test2"][1].values["@module_type"])
            dump = proc.dumpPython()
            self.assertEqual(dump.find('@'), -1)
            self.assertEqual(specialImportRegistry.getSpecialImports(), ["from test import SwitchProducerTest"])
            self.assertTrue(dump.find("\nfrom test import SwitchProducerTest\n") != -1)

            # EDAlias as non-chosen case
            proc = Process("test")
            proc.sp = SwitchProducerTest(test2 = EDProducer("Foo",
                                                            a = int32(1),
                                                            b = PSet(c = int32(2))),
                                         test1 = EDAlias(a = VPSet(PSet(type = string("Bar")))))
            proc.a = EDProducer("A")
            proc.s = Sequence(proc.a + proc.sp)
            proc.t = Task(proc.a, proc.sp)
            proc.p = Path()
            proc.p.associate(proc.t)
            p = TestMakePSet()
            proc.fillProcessDesc(p)
            self.assertEqual((True,"EDProducer"), p.values["sp"][1].values["@module_edm_type"])
            self.assertEqual((True, "SwitchProducer"), p.values["sp"][1].values["@module_type"])
            self.assertEqual((True, "sp"), p.values["sp"][1].values["@module_label"])
            all_cases = copy.deepcopy(p.values["sp"][1].values["@all_cases"])
            all_cases[1].sort()
            self.assertEqual((True, ["sp@test1", "sp@test2"]), all_cases)
            self.assertEqual((False, "sp@test2"), p.values["sp"][1].values["@chosen_case"])
            self.assertEqual(["a", "sp", "sp@test2"], p.values["@all_modules"][1])
            self.assertEqual(["sp@test1"], p.values["@all_aliases"][1])
            self.assertEqual((True,"EDProducer"), p.values["sp@test2"][1].values["@module_edm_type"])
            self.assertEqual((True,"Foo"), p.values["sp@test2"][1].values["@module_type"])
            self.assertEqual((True,"EDAlias"), p.values["sp@test1"][1].values["@module_edm_type"])
            self.assertEqual((True,"Bar"), p.values["sp@test1"][1].values["a"][1][0].values["type"])

            # EDAlias as chosen case
            proc = Process("test")
            proc.sp = SwitchProducerTest(test1 = EDProducer("Foo",
                                                            a = int32(1),
                                                            b = PSet(c = int32(2))),
                                         test2 = EDAlias(a = VPSet(PSet(type = string("Bar")))))
            proc.a = EDProducer("A")
            proc.s = Sequence(proc.a + proc.sp)
            proc.t = Task(proc.a, proc.sp)
            proc.p = Path()
            proc.p.associate(proc.t)
            p = TestMakePSet()
            proc.fillProcessDesc(p)
            self.assertEqual((True,"EDProducer"), p.values["sp"][1].values["@module_edm_type"])
            self.assertEqual((True, "SwitchProducer"), p.values["sp"][1].values["@module_type"])
            self.assertEqual((True, "sp"), p.values["sp"][1].values["@module_label"])
            self.assertEqual((True, ["sp@test1", "sp@test2"]), p.values["sp"][1].values["@all_cases"])
            self.assertEqual((False, "sp@test2"), p.values["sp"][1].values["@chosen_case"])
            self.assertEqual(["a", "sp", "sp@test1"], p.values["@all_modules"][1])
            self.assertEqual(["sp@test2"], p.values["@all_aliases"][1])
            self.assertEqual((True,"EDProducer"), p.values["sp@test1"][1].values["@module_edm_type"])
            self.assertEqual((True,"Foo"), p.values["sp@test1"][1].values["@module_type"])
            self.assertEqual((True,"EDAlias"), p.values["sp@test2"][1].values["@module_edm_type"])
            self.assertEqual((True,"Bar"), p.values["sp@test2"][1].values["a"][1][0].values["type"])

        def testPrune(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.d = EDAnalyzer("OurAnalyzer")
            p.e = EDProducer("MyProducer")
            p.f = EDProducer("YourProducer")
            p.g = EDProducer("TheirProducer")
            p.s = Sequence(p.d)
            p.t1 = Task(p.e)
            p.t2 = Task(p.f)
            p.t3 = Task(p.g, p.t1)
            p.path1 = Path(p.a, p.t3)
            p.path2 = Path(p.b)
            self.assertTrue(p.schedule is None)
            pths = p.paths
            keys = pths.keys()
            self.assertEqual(pths[keys[0]],p.path1)
            self.assertEqual(pths[keys[1]],p.path2)
            p.pset1 = PSet(parA = string("pset1"))
            p.pset2 = untracked.PSet(parA = string("pset2"))
            p.vpset1 = VPSet()
            p.vpset2 = untracked.VPSet()
            p.prune()
            self.assertTrue(hasattr(p, 'a'))
            self.assertTrue(hasattr(p, 'b'))
            self.assertTrue(not hasattr(p, 'c'))
            self.assertTrue(not hasattr(p, 'd'))
            self.assertTrue(hasattr(p, 'e'))
            self.assertTrue(not hasattr(p, 'f'))
            self.assertTrue(hasattr(p, 'g'))
            self.assertTrue(not hasattr(p, 's'))
            self.assertTrue(hasattr(p, 't1'))
            self.assertTrue(not hasattr(p, 't2'))
            self.assertTrue(hasattr(p, 't3'))
            self.assertTrue(hasattr(p, 'path1'))
            self.assertTrue(hasattr(p, 'path2'))
#            self.assertTrue(not hasattr(p, 'pset1'))
#            self.assertTrue(hasattr(p, 'pset2'))
#            self.assertTrue(not hasattr(p, 'vpset1'))
#            self.assertTrue(not hasattr(p, 'vpset2'))

            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.d = EDAnalyzer("OurAnalyzer")
            p.e = EDAnalyzer("OurAnalyzer")
            p.f = EDProducer("MyProducer")
            p.g = EDProducer("YourProducer")
            p.h = EDProducer("TheirProducer")
            p.i = EDProducer("OurProducer")
            p.t1 = Task(p.f)
            p.t2 = Task(p.g)
            p.t3 = Task(p.h)
            p.t4 = Task(p.i)
            p.s = Sequence(p.d, p.t1)
            p.s2 = Sequence(p.b, p.t2)
            p.s3 = Sequence(p.e)
            p.path1 = Path(p.a, p.t3)
            p.path2 = Path(p.b)
            p.path3 = Path(p.b+p.s2)
            p.path4 = Path(p.b+p.s3)
            p.schedule = Schedule(p.path1,p.path2,p.path3)
            p.schedule.associate(p.t4)
            pths = p.paths
            keys = pths.keys()
            self.assertEqual(pths[keys[0]],p.path1)
            self.assertEqual(pths[keys[1]],p.path2)
            p.prune()
            self.assertTrue(hasattr(p, 'a'))
            self.assertTrue(hasattr(p, 'b'))
            self.assertTrue(not hasattr(p, 'c'))
            self.assertTrue(not hasattr(p, 'd'))
            self.assertTrue(not hasattr(p, 'e'))
            self.assertTrue(not hasattr(p, 'f'))
            self.assertTrue(hasattr(p, 'g'))
            self.assertTrue(hasattr(p, 'h'))
            self.assertTrue(hasattr(p, 'i'))
            self.assertTrue(not hasattr(p, 't1'))
            self.assertTrue(hasattr(p, 't2'))
            self.assertTrue(hasattr(p, 't3'))
            self.assertTrue(hasattr(p, 't4'))
            self.assertTrue(not hasattr(p, 's'))
            self.assertTrue(hasattr(p, 's2'))
            self.assertTrue(not hasattr(p, 's3'))
            self.assertTrue(hasattr(p, 'path1'))
            self.assertTrue(hasattr(p, 'path2'))
            self.assertTrue(hasattr(p, 'path3'))
            self.assertTrue(not hasattr(p, 'path4'))
            #test SequencePlaceholder
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.s = Sequence(SequencePlaceholder("a")+p.b)
            p.pth = Path(p.s)
            p.prune()
            self.assertTrue(hasattr(p, 'a'))
            self.assertTrue(hasattr(p, 'b'))
            self.assertTrue(hasattr(p, 's'))
            self.assertTrue(hasattr(p, 'pth'))
            #test unresolved SequencePlaceholder
            p = Process("test")
            p.b = EDAnalyzer("YourAnalyzer")
            p.s = Sequence(SequencePlaceholder("a")+p.b)
            p.pth = Path(p.s)
            p.prune(keepUnresolvedSequencePlaceholders=True)
            self.assertTrue(hasattr(p, 'b'))
            self.assertTrue(hasattr(p, 's'))
            self.assertTrue(hasattr(p, 'pth'))
            self.assertEqual(p.s.dumpPython(),'cms.Sequence(cms.SequencePlaceholder("a")+process.b)\n')
            #test TaskPlaceholder
            p = Process("test")
            p.a = EDProducer("MyProducer")
            p.b = EDProducer("YourProducer")
            p.s = Task(TaskPlaceholder("a"),p.b)
            p.pth = Path(p.s)
            p.prune()
            self.assertTrue(hasattr(p, 'a'))
            self.assertTrue(hasattr(p, 'b'))
            self.assertTrue(hasattr(p, 's'))
            self.assertTrue(hasattr(p, 'pth'))
            #test unresolved SequencePlaceholder
            p = Process("test")
            p.b = EDProducer("YourAnalyzer")
            p.s = Task(TaskPlaceholder("a"),p.b)
            p.pth = Path(p.s)
            p.prune(keepUnresolvedSequencePlaceholders=True)
            self.assertTrue(hasattr(p, 'b'))
            self.assertTrue(hasattr(p, 's'))
            self.assertTrue(hasattr(p, 'pth'))
            self.assertEqual(p.s.dumpPython(),'cms.Task(cms.TaskPlaceholder("a"), process.b)\n')
        def testTaskPlaceholder(self):
            p = Process("test")
            p.a = EDProducer("ma")
            p.b = EDAnalyzer("mb")
            p.t1 = Task(TaskPlaceholder("c"))
            p.t2 = Task(p.a, TaskPlaceholder("d"), p.t1)
            p.t3 = Task(TaskPlaceholder("e"))
            p.path1 = Path(p.b, p.t2, p.t3)
            p.t5 = Task(p.a, TaskPlaceholder("g"), TaskPlaceholder("t4"))
            p.t4 = Task(TaskPlaceholder("f"))
            p.endpath1 = EndPath(p.b, p.t5)
            p.t6 = Task(TaskPlaceholder("h"))
            p.t7 = Task(p.a, TaskPlaceholder("i"), p.t6)
            p.t8 = Task(TaskPlaceholder("j"))
            p.schedule = Schedule(p.path1, p.endpath1,tasks=[p.t7,p.t8])
            p.c = EDProducer("mc")
            p.d = EDProducer("md")
            p.e = EDProducer("me")
            p.f = EDProducer("mf")
            p.g = EDProducer("mg")
            p.h = EDProducer("mh")
            p.i = EDProducer("mi")
            p.j = EDProducer("mj")
            self.assertEqual(_lineDiff(p.dumpPython(),Process('test').dumpPython()),
"""process.a = cms.EDProducer("ma")
process.c = cms.EDProducer("mc")
process.d = cms.EDProducer("md")
process.e = cms.EDProducer("me")
process.f = cms.EDProducer("mf")
process.g = cms.EDProducer("mg")
process.h = cms.EDProducer("mh")
process.i = cms.EDProducer("mi")
process.j = cms.EDProducer("mj")
process.b = cms.EDAnalyzer("mb")
process.t8 = cms.Task(cms.TaskPlaceholder("j"))
process.t6 = cms.Task(cms.TaskPlaceholder("h"))
process.t7 = cms.Task(cms.TaskPlaceholder("i"), process.a, process.t6)
process.t4 = cms.Task(cms.TaskPlaceholder("f"))
process.t5 = cms.Task(cms.TaskPlaceholder("g"), cms.TaskPlaceholder("t4"), process.a)
process.t3 = cms.Task(cms.TaskPlaceholder("e"))
process.t1 = cms.Task(cms.TaskPlaceholder("c"))
process.t2 = cms.Task(cms.TaskPlaceholder("d"), process.a, process.t1)
process.path1 = cms.Path(process.b, process.t2, process.t3)
process.endpath1 = cms.EndPath(process.b, process.t5)
process.schedule = cms.Schedule(*[ process.path1, process.endpath1 ], tasks=[process.t7, process.t8])""")
            p.resolve()
            self.assertEqual(_lineDiff(p.dumpPython(),Process('test').dumpPython()),
"""process.a = cms.EDProducer("ma")
process.c = cms.EDProducer("mc")
process.d = cms.EDProducer("md")
process.e = cms.EDProducer("me")
process.f = cms.EDProducer("mf")
process.g = cms.EDProducer("mg")
process.h = cms.EDProducer("mh")
process.i = cms.EDProducer("mi")
process.j = cms.EDProducer("mj")
process.b = cms.EDAnalyzer("mb")
process.t8 = cms.Task(process.j)
process.t6 = cms.Task(process.h)
process.t7 = cms.Task(process.a, process.i, process.t6)
process.t4 = cms.Task(process.f)
process.t5 = cms.Task(process.a, process.g, process.t4)
process.t3 = cms.Task(process.e)
process.t1 = cms.Task(process.c)
process.t2 = cms.Task(process.a, process.d, process.t1)
process.path1 = cms.Path(process.b, process.t2, process.t3)
process.endpath1 = cms.EndPath(process.b, process.t5)
process.schedule = cms.Schedule(*[ process.path1, process.endpath1 ], tasks=[process.t7, process.t8])""")

        def testDelete(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.d = EDAnalyzer("OurAnalyzer")
            p.e = EDAnalyzer("OurAnalyzer")
            p.f = EDAnalyzer("OurAnalyzer")
            p.g = EDProducer("OurProducer")
            p.h = EDProducer("YourProducer")
            p.t1 = Task(p.g, p.h)
            t2 = Task(p.g, p.h)
            t3 = Task(p.g, p.h)
            p.s = Sequence(p.d+p.e)
            p.path1 = Path(p.a+p.f+p.s,t2)
            p.endpath1 = EndPath(p.b+p.f)
            p.schedule = Schedule(tasks=[t3])
            self.assertTrue(hasattr(p, 'f'))
            self.assertTrue(hasattr(p, 'g'))
            del p.e
            del p.f
            del p.g
            self.assertFalse(hasattr(p, 'f'))
            self.assertFalse(hasattr(p, 'g'))
            self.assertTrue(p.t1.dumpPython() == 'cms.Task(process.h)\n')
            self.assertTrue(p.s.dumpPython() == 'cms.Sequence(process.d)\n')
            self.assertTrue(p.path1.dumpPython() == 'cms.Path(process.a+process.s, cms.Task(process.h))\n')
            self.assertTrue(p.endpath1.dumpPython() == 'cms.EndPath(process.b)\n')
            del p.s
            self.assertTrue(p.path1.dumpPython() == 'cms.Path(process.a+(process.d), cms.Task(process.h))\n')
            self.assertTrue(p.schedule_().dumpPython() == 'cms.Schedule(tasks=[cms.Task(process.h)])\n')
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
            self.assertTrue(p.isUsingModifier(m1))
            #check that Modifier not attached to a process doesn't run
            m1 = Modifier()
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1))
            m1.toModify(p.a,_mod_fred)
            p.b = EDAnalyzer("YourAnalyzer", wilma = int32(1))
            m1.toModify(p.b, wilma = 2)
            self.assertEqual(p.a.fred.value(),1)
            self.assertEqual(p.b.wilma.value(),1)
            self.assertEqual(p.isUsingModifier(m1),False)
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
            #test removal of parameter
            m1 = Modifier()
            p = Process("test",m1)
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1), wilma = int32(1), fintstones = PSet(fred = int32(1)))
            m1.toModify(p.a, fred = None, fintstones = dict(fred = None))
            self.assertEqual(hasattr(p.a, "fred"), False)
            self.assertEqual(hasattr(p.a.fintstones, "fred"), False)
            self.assertEqual(p.a.wilma.value(),1)
            #test adding a parameter
            m1 = Modifier()
            p = Process("test",m1)
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1))
            m1.toModify(p.a, wilma = int32(2))
            self.assertEqual(p.a.fred.value(), 1)
            self.assertEqual(p.a.wilma.value(),2)
            #test setting of value in PSet
            m1 = Modifier()
            p = Process("test",m1)
            p.a = EDAnalyzer("MyAnalyzer", flintstones = PSet(fred = int32(1), wilma = int32(1)))
            m1.toModify(p.a, flintstones = dict(fred = int32(2)))
            self.assertEqual(p.a.flintstones.fred.value(),2)
            self.assertEqual(p.a.flintstones.wilma.value(),1)
            #test proper exception from nonexisting parameter name
            m1 = Modifier()
            p = Process("test",m1)
            p.a = EDAnalyzer("MyAnalyzer", flintstones = PSet(fred = PSet(wilma = int32(1))))
            self.assertRaises(KeyError, lambda: m1.toModify(p.a, flintstones = dict(imnothere = dict(wilma=2))))
            self.assertRaises(KeyError, lambda: m1.toModify(p.a, foo = 1))
            #test setting a value in a VPSet
            m1 = Modifier()
            p = Process("test",m1)
            p.a = EDAnalyzer("MyAnalyzer", flintstones = VPSet(PSet(fred = int32(1)), PSet(wilma = int32(1))))
            m1.toModify(p.a, flintstones = {1:dict(wilma = int32(2))})
            self.assertEqual(p.a.flintstones[0].fred.value(),1)
            self.assertEqual(p.a.flintstones[1].wilma.value(),2)
            #test setting a value in a list of values
            m1 = Modifier()
            p = Process("test",m1)
            p.a = EDAnalyzer("MyAnalyzer", fred = vuint32(1,2,3))
            m1.toModify(p.a, fred = {1:7})
            self.assertEqual(p.a.fred[0],1)
            self.assertEqual(p.a.fred[1],7)
            self.assertEqual(p.a.fred[2],3)
            #test IndexError setting a value in a list to an item key not in the list
            m1 = Modifier()
            p = Process("test",m1)
            p.a = EDAnalyzer("MyAnalyzer", fred = vuint32(1,2,3))
            raised = False
            try: m1.toModify(p.a, fred = {5:7})
            except IndexError as e: raised = True
            self.assertEqual(raised, True)
            #test TypeError setting a value in a list using a key that is not an int
            m1 = Modifier()
            p = Process("test",m1)
            p.a = EDAnalyzer("MyAnalyzer", flintstones = VPSet(PSet(fred = int32(1)), PSet(wilma = int32(1))))
            raised = False
            try: m1.toModify(p.a, flintstones = dict(bogus = int32(37)))
            except TypeError as e: raised = True
            self.assertEqual(raised, True)
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
            self.assertTrue(hasattr(p,"a"))
            m1 = Modifier()
            p = Process("test",m1)
            testProcMod = ProcModifierMod(m1,_rem_a)
            p.extend(testMod)
            p.extend(testProcMod)
            self.assertTrue(not hasattr(p,"a"))
            #test ModifierChain
            m1 = Modifier()
            mc = ModifierChain(m1)
            p = Process("test",mc)
            self.assertTrue(p.isUsingModifier(m1))
            self.assertTrue(p.isUsingModifier(mc))
            testMod = DummyMod()
            p.b = EDAnalyzer("Dummy2", fred = int32(1))
            m1.toModify(p.b, fred = int32(3))
            p.extend(testMod)
            testProcMod = ProcModifierMod(m1,_rem_a)
            p.extend(testProcMod)
            self.assertTrue(not hasattr(p,"a"))
            self.assertEqual(p.b.fred.value(),3)
            #check cloneAndExclude
            m1 = Modifier()
            m2 = Modifier()
            mc = ModifierChain(m1,m2)
            mclone = mc.copyAndExclude([m2])
            self.assertTrue(not mclone._isOrContains(m2))
            self.assertTrue(mclone._isOrContains(m1))
            m3 = Modifier()
            mc2 = ModifierChain(mc,m3)
            mclone = mc2.copyAndExclude([m2])
            self.assertTrue(not mclone._isOrContains(m2))
            self.assertTrue(mclone._isOrContains(m1))
            self.assertTrue(mclone._isOrContains(m3))
            #check combining
            m1 = Modifier()
            m2 = Modifier()
            p = Process("test",m1)
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1), wilma = int32(1))
            (m1 & m2).toModify(p.a, fred = int32(2))
            self.assertRaises(TypeError, lambda: (m1 & m2).toModify(p.a, 1, wilma=2))
            self.assertEqual(p.a.fred, 1)
            m1 = Modifier()
            m2 = Modifier()
            p = Process("test",m1,m2)
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1), wilma = int32(1))
            (m1 & m2).toModify(p.a, fred = int32(2))
            self.assertEqual(p.a.fred, 2)
            m1 = Modifier()
            m2 = Modifier()
            m3 = Modifier()
            p = Process("test",m1,m2,m3)
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1), wilma = int32(1))
            (m1 & m2 & m3).toModify(p.a, fred = int32(2))
            self.assertEqual(p.a.fred, 2)
            (m1 & (m2 & m3)).toModify(p.a, fred = int32(3))
            self.assertEqual(p.a.fred, 3)
            ((m1 & m2) & m3).toModify(p.a, fred = int32(4))
            self.assertEqual(p.a.fred, 4)
            #check inverse
            m1 = Modifier()
            m2 = Modifier()
            p = Process("test", m1)
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1), wilma = int32(1))
            (~m1).toModify(p.a, fred=2)
            self.assertEqual(p.a.fred, 1)
            (~m2).toModify(p.a, wilma=2)
            self.assertEqual(p.a.wilma, 2)
            self.assertRaises(TypeError, lambda: (~m1).toModify(p.a, 1, wilma=2))
            self.assertRaises(TypeError, lambda: (~m2).toModify(p.a, 1, wilma=2))
            # check or
            m1 = Modifier()
            m2 = Modifier()
            m3 = Modifier()
            p = Process("test", m1)
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1), wilma = int32(1))
            (m1 | m2).toModify(p.a, fred=2)
            self.assertEqual(p.a.fred, 2)
            (m1 | m2 | m3).toModify(p.a, fred=3)
            self.assertEqual(p.a.fred, 3)
            (m3 | m2 | m1).toModify(p.a, fred=4)
            self.assertEqual(p.a.fred, 4)
            ((m1 | m2) | m3).toModify(p.a, fred=5)
            self.assertEqual(p.a.fred, 5)
            (m1 | (m2 | m3)).toModify(p.a, fred=6)
            self.assertEqual(p.a.fred, 6)
            (m2 | m3).toModify(p.a, fred=7)
            self.assertEqual(p.a.fred, 6)
            self.assertRaises(TypeError, lambda: (m1 | m2).toModify(p.a, 1, wilma=2))
            self.assertRaises(TypeError, lambda: (m2 | m3).toModify(p.a, 1, wilma=2))
            # check combinations
            m1 = Modifier()
            m2 = Modifier()
            m3 = Modifier()
            m4 = Modifier()
            p = Process("test", m1, m2)
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1), wilma = int32(1))
            (m1 & ~m2).toModify(p.a, fred=2)
            self.assertEqual(p.a.fred, 1)
            (m1 & ~m3).toModify(p.a, fred=2)
            self.assertEqual(p.a.fred, 2)
            (m1 | ~m2).toModify(p.a, fred=3)
            self.assertEqual(p.a.fred, 3)
            (~m1 | ~m2).toModify(p.a, fred=4)
            self.assertEqual(p.a.fred, 3)
            (~m3 & ~m4).toModify(p.a, fred=4)
            self.assertEqual(p.a.fred, 4)
            ((m1 & m3) | ~m4).toModify(p.a, fred=5)
            self.assertEqual(p.a.fred, 5)
            #check toReplaceWith
            m1 = Modifier()
            p = Process("test",m1)
            p.a =EDAnalyzer("MyAnalyzer", fred = int32(1))
            m1.toReplaceWith(p.a, EDAnalyzer("YourAnalyzer", wilma = int32(3)))
            self.assertRaises(TypeError, lambda: m1.toReplaceWith(p.a, EDProducer("YourProducer")))
            p.b =EDAnalyzer("BAn")
            p.c =EDProducer("c")
            p.d =EDProducer("d")
            p.tc = Task(p.c)
            p.td = Task(p.d)
            p.s = Sequence(p.a, p.tc)
            m1.toReplaceWith(p.s, Sequence(p.a+p.b, p.td))
            self.assertEqual(p.a.wilma.value(),3)
            self.assertEqual(p.a.type_(),"YourAnalyzer")
            self.assertEqual(hasattr(p,"fred"),False)
            self.assertTrue(p.s.dumpPython() == "cms.Sequence(process.a+process.b, process.td)\n")
            p.e =EDProducer("e")
            m1.toReplaceWith(p.td, Task(p.e))
            self.assertTrue(p.td._collection == OrderedSet([p.e]))
            #check toReplaceWith doesn't activate not chosen
            m1 = Modifier()
            p = Process("test")
            p.a =EDAnalyzer("MyAnalyzer", fred = int32(1))
            m1.toReplaceWith(p.a, EDAnalyzer("YourAnalyzer", wilma = int32(3)))
            self.assertEqual(p.a.type_(),"MyAnalyzer")
            #check toReplaceWith and and/not/or combinations
            m1 = Modifier()
            m2 = Modifier()
            m3 = Modifier()
            m4 = Modifier()
            p = Process("test", m1, m2)
            p.a = EDAnalyzer("MyAnalyzer", fred = int32(1), wilma = int32(1))
            self.assertRaises(TypeError, lambda: (m1 & m2).toReplaceWith(p.a, EDProducer("YourProducer")))
            self.assertRaises(TypeError, lambda: (m3 & m4).toReplaceWith(p.a, EDProducer("YourProducer")))
            self.assertRaises(TypeError, lambda: (~m3).toReplaceWith(p.a, EDProducer("YourProducer")))
            self.assertRaises(TypeError, lambda: (~m1).toReplaceWith(p.a, EDProducer("YourProducer")))
            self.assertRaises(TypeError, lambda: (m1 | m3).toReplaceWith(p.a, EDProducer("YourProducer")))
            self.assertRaises(TypeError, lambda: (m3 | m4).toReplaceWith(p.a, EDProducer("YourProducer")))
            (m1 & m2).toReplaceWith(p.a, EDAnalyzer("YourAnalyzer1"))
            self.assertEqual(p.a.type_(), "YourAnalyzer1")
            (m1 & m3).toReplaceWith(p.a, EDAnalyzer("YourAnalyzer2"))
            self.assertEqual(p.a.type_(), "YourAnalyzer1")
            (~m1).toReplaceWith(p.a, EDAnalyzer("YourAnalyzer2"))
            self.assertEqual(p.a.type_(), "YourAnalyzer1")
            (~m3).toReplaceWith(p.a, EDAnalyzer("YourAnalyzer2"))
            self.assertEqual(p.a.type_(), "YourAnalyzer2")
            (m1 | m3).toReplaceWith(p.a, EDAnalyzer("YourAnalyzer3"))
            self.assertEqual(p.a.type_(), "YourAnalyzer3")
            (m3 | m4).toReplaceWith(p.a, EDAnalyzer("YourAnalyzer4"))
            self.assertEqual(p.a.type_(), "YourAnalyzer3")

            # EDAlias
            a = EDAlias(foo2 = VPSet(PSet(type = string("Foo2"))))
            m = Modifier()
            m._setChosen()
            # Modify parameters
            m.toModify(a, foo2 = {0: dict(type = "Foo3")})
            self.assertEqual(a.foo2[0].type, "Foo3")
            # Add an alias
            m.toModify(a, foo4 = VPSet(PSet(type = string("Foo4"))))
            self.assertEqual(a.foo2[0].type, "Foo3")
            self.assertEqual(a.foo4[0].type, "Foo4")
            # Remove an alias
            m.toModify(a, foo2 = None)
            self.assertFalse(hasattr(a, "foo2"))
            self.assertEqual(a.foo4[0].type, "Foo4")
            # Replace (doesn't work out of the box because EDAlias is not _Parameterizable
            m.toReplaceWith(a, EDAlias(bar = VPSet(PSet(type = string("Bar")))))
            self.assertFalse(hasattr(a, "foo2"))
            self.assertFalse(hasattr(a, "foo4"))
            self.assertTrue(hasattr(a, "bar"))
            self.assertEqual(a.bar[0].type, "Bar")

            # SwitchProducer
            sp = SwitchProducerTest(test1 = EDProducer("Foo",
                                                       a = int32(1),
                                                       b = PSet(c = int32(2))),
                                    test2 = EDProducer("Bar",
                                                       aa = int32(11),
                                                       bb = PSet(cc = int32(12))))
            m = Modifier()
            m._setChosen()
            # Modify parameters
            m.toModify(sp,
                       test1 = dict(a = 4, b = dict(c = None)),
                       test2 = dict(aa = 15, bb = dict(cc = 45, dd = string("foo"))))
            self.assertEqual(sp.test1.a.value(), 4)
            self.assertEqual(sp.test1.b.hasParameter("c"), False)
            self.assertEqual(sp.test2.aa.value(), 15)
            self.assertEqual(sp.test2.bb.cc.value(), 45)
            self.assertEqual(sp.test2.bb.dd.value(), "foo")
            # Replace a producer
            m.toReplaceWith(sp.test1, EDProducer("Fred", x = int32(42)))
            self.assertEqual(sp.test1.type_(), "Fred")
            self.assertEqual(sp.test1.x.value(), 42)
            self.assertRaises(TypeError, lambda: m.toReplaceWith(sp.test1, EDAnalyzer("Foo")))
            # Alternative way (only to be allow same syntax to be used as for adding)
            m.toModify(sp, test2 = EDProducer("Xyzzy", x = int32(24)))
            self.assertEqual(sp.test2.type_(), "Xyzzy")
            self.assertEqual(sp.test2.x.value(), 24)
            self.assertRaises(TypeError, lambda: m.toModify(sp, test2 = EDAnalyzer("Foo")))
            # Add a producer
            m.toModify(sp, test3 = EDProducer("Wilma", y = int32(24)))
            self.assertEqual(sp.test3.type_(), "Wilma")
            self.assertEqual(sp.test3.y.value(), 24)
            self.assertRaises(TypeError, lambda: m.toModify(sp, test4 = EDAnalyzer("Foo")))
            # Remove a producer
            m.toModify(sp, test2 = None)
            self.assertEqual(hasattr(sp, "test2"), False)
            # Add an alias
            m.toModify(sp, test2 = EDAlias(foo = VPSet(PSet(type = string("int")))))
            self.assertTrue(hasattr(sp.test2, "foo"))
            # Replace an alias
            m.toReplaceWith(sp.test2, EDAlias(bar = VPSet(PSet(type = string("int")))))
            self.assertTrue(hasattr(sp.test2, "bar"))
            # Alternative way
            m.toModify(sp, test2 = EDAlias(xyzzy = VPSet(PSet(type = string("int")))))
            self.assertTrue(hasattr(sp.test2, "xyzzy"))
            # Replace an alias with EDProducer
            self.assertRaises(TypeError, lambda: m.toReplaceWith(sp.test2, EDProducer("Foo")))
            m.toModify(sp, test2 = EDProducer("Foo"))
        def testProcessFragment(self):
            #check defaults are not overwritten
            f = ProcessFragment('Fragment')
            p = Process('PROCESS')
            p.maxEvents.input = 10
            p.options.numberOfThreads = 4
            p.maxLuminosityBlocks.input = 2
            p.extend(f)
            self.assertEqual(p.maxEvents.input.value(),10)
            self.assertEqual(p.options.numberOfThreads.value(), 4)
            self.assertEqual(p.maxLuminosityBlocks.input.value(),2)
            #general checks
            f = ProcessFragment("Fragment")
            f.fltr = EDFilter("Foo")
            p = Process('PROCESS')
            p.extend(f)
            self.assertTrue(hasattr(p,'fltr'))

    unittest.main()
