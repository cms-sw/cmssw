#!/usr/bin/env python

#command line options helper
from  Options import Options
options = Options()

from Mixins import _SimpleParameterTypeBase, _ParameterTypeBase, _Parameterizable, _ConfigureComponent, _TypedParameterizable
from Mixins import  _Labelable,  _Unlabelable 
from Mixins import _ValidatingListBase
from Types import * 

# helper classes for sorted and fixed dicts
class SortedKeysDict(dict):
    """a dict preserving order of keys"""
    # specialised __repr__ missing.
    def __init__(self,*args,**kw):
        dict.__init__(self,*args,**kw)
        self.list = list()
        if len(args) == 1:
            if not hasattr(args[0],'iterkeys'):
                self.list= [ x[0] for x in iter(args[0])]
            else:
                self.list = list(args[0].iterkeys())
            return
        self.list = list(super(SortedKeysDict,self).iterkeys())
    def __iter__(self):
        for key in self.list:
            yield key

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        if not key in self.list:
            self.list.append(key)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self.list.remove(key)

    def items(self):
        return [ dict.__getitem__(self, key) for key in self.list]

    def iteritems(self):
        for key in self.list:
            yield key, dict.__getitem__(self, key)

    def iterkeys(self):
        for key in self.list:
            yield key
           
    def itervalues(self):
        for key in self.list:
            yield dict.__getitem__(self,key)

    def keys(self):
        return self.list

    def values(self):
        return [ dict.__getitems__(self, key) for key in self.list]

   
class SortedAndFixedKeysDict(SortedKeysDict):
    """a sorted dictionary with fixed/frozen keys"""
    def _blocked_attribute(obj):
        raise AttributeError, "A SortedAndFixedKeysDict cannot be modified."
    _blocked_attribute = property(_blocked_attribute)
    __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute
    def __new__(cls, *args, **kw):
        new = SortedKeysDict.__new__(cls)
        SortedKeysDict.__init__(new, *args, **kw)
        return new
    def __init__(self, *args, **kw):
        pass
    def __repr__(self):
        return "SortedAndFixedKeysDict(%s)" % SortedKeysDict.__repr__(self)

    
#helper based on code from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/414283
class FixedKeysDict(dict):
    def _blocked_attribute(obj):
        raise AttributeError, "A FixedKeysDict cannot be modified."
    _blocked_attribute = property(_blocked_attribute)

    __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute
    def __new__(cls, *args, **kw):
        new = dict.__new__(cls)
        dict.__init__(new, *args, **kw)
        return new
    def __init__(self, *args, **kw):
        pass
    def __repr__(self):
        return "FixedKeysDict(%s)" % dict.__repr__(self)

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
    def __init__(self,name):
        self.__dict__['_Process__name'] = name
        self.__dict__['_Process__filters'] = {}
        self.__dict__['_Process__producers'] = {}
        self.__dict__['_Process__source'] = None
        self.__dict__['_Process__looper'] = None
        self.__dict__['_Process__schedule'] = None
        self.__dict__['_Process__analyzers'] = {}
        self.__dict__['_Process__outputmodules'] = {}
        self.__dict__['_Process__paths'] = SortedKeysDict()    # have to keep the order
        self.__dict__['_Process__endpaths'] = SortedKeysDict() # of definition
        self.__dict__['_Process__sequences'] = {}
        self.__dict__['_Process__services'] = {}
        self.__dict__['_Process__essources'] = {}
        self.__dict__['_Process__esproducers'] = {}
        self.__dict__['_Process__esprefers'] = {}
        self.__dict__['_Process__psets']={}
        self.__dict__['_Process__vpsets']={}
        self.__dict__['_cloneToObjectDict'] = {}
    def filters_(self):
        """returns a dict of the filters which have been added to the Process"""
        return FixedKeysDict(self.__filters)
    filters = property(filters_, doc="dictionary containing the filters for the process")
    def name_(self):
        return self.__name
    def setName_(self,name):
        self.__name = name
    process = property(name_,setName_, doc="name of the process")
    def producers_(self):
        """returns a dict of the producers which have been added to the Process"""
        return FixedKeysDict(self.__producers)
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
    def analyzers_(self):
        """returns a dict of the filters which have been added to the Process"""
        return FixedKeysDict(self.__analyzers)
    analyzers = property(analyzers_,doc="dictionary containing the analyzers for the process")
    def outputModules_(self):
        """returns a dict of the output modules which have been added to the Process"""
        return FixedKeysDict(self.__outputmodules)
    outputModules = property(outputModules_,doc="dictionary containing the output_modules for the process")
    def paths_(self):
        """returns a dict of the paths which have been added to the Process"""
        return SortedAndFixedKeysDict(self.__paths)
    paths = property(paths_,doc="dictionary containing the paths for the process")
    def endpaths_(self):
        """returns a dict of the endpaths which have been added to the Process"""
        return SortedAndFixedKeysDict(self.__endpaths)
    endpaths = property(endpaths_,doc="dictionary containing the endpaths for the process")
    def sequences_(self):
        """returns a dict of the sequences which have been added to the Process"""
        return FixedKeysDict(self.__sequences)
    sequences = property(sequences_,doc="dictionary containing the sequences for the process")
    def schedule_(self):
        """returns the schedule which has been added to the Process or None if none have been added"""
        return self.__schedule
    def setSchedule_(self,sch):
        self.__dict__['_Process__schedule'] = sch
    schedule = property(schedule_,setSchedule_,doc='the schedule or None if not set')
    def services_(self):
        """returns a dict of the services which have been added to the Process"""
        return FixedKeysDict(self.__services)
    services = property(services_,doc="dictionary containing the services for the process")
    def es_producers_(self):
        """returns a dict of the esproducers which have been added to the Process"""
        return FixedKeysDict(self.__esproducers)
    es_producers = property(es_producers_,doc="dictionary containing the es_producers for the process")
    def es_sources_(self):
        """returns a the es_sources which have been added to the Process"""
        return FixedKeysDict(self.__essources)
    es_sources = property(es_sources_,doc="dictionary containing the es_sources for the process")
    def es_prefers_(self):
        """returns a dict of the es_prefers which have been added to the Process"""
        return FixedKeysDict(self.__esprefers)
    es_prefers = property(es_prefers_,doc="dictionary containing the es_prefers for the process")
    def psets_(self):
        """returns a dict of the PSets which have been added to the Process"""
        return FixedKeysDict(self.__psets)
    psets = property(psets_,doc="dictionary containing the PSets for the process")
    def vpsets_(self):
        """returns a dict of the VPSets which have been added to the Process"""
        return FixedKeysDict(self.__vpsets)
    vpsets = property(vpsets_,doc="dictionary containing the PSets for the process")
    def __setattr__(self,name,value):
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
        newValue =value.copy()

        self.__dict__[name]=newValue
        if isinstance(newValue,_Labelable):
            newValue.setLabel(name)
            self._cloneToObjectDict[id(value)] = newValue
            self._cloneToObjectDict[id(newValue)] = newValue
        #now put in proper bucket
        newValue._place(name,self)
        
    def __delattr__(self,name):
        pass

    def add_(self,value):
        """Allows addition of components which do not have to have a label, e.g. Services"""
        if not isinstance(value,_ConfigureComponent):
            raise TypeError
        if not isinstance(value,_Unlabelable):
            raise TypeError
        #clone the item
        newValue =value.copy()
        newValue._place('',self)
        
    def _placeOutputModule(self,name,mod):
        self.__outputmodules[name]=mod
    def _placeProducer(self,name,mod):
        self.__producers[name]=mod
    def _placeFilter(self,name,mod):
        self.__filters[name]=mod
    def _placeAnalyzer(self,name,mod):
        self.__analyzers[name]=mod
    def _placePath(self,name,mod):
        self.__paths[name]=mod._clonesequence(self._cloneToObjectDict)
    def _placeEndPath(self,name,mod):
        self.__endpaths[name]=mod._clonesequence(self._cloneToObjectDict)
    def _placeSequence(self,name,mod):
        self.__sequences[name]=mod._clonesequence(self._cloneToObjectDict)
    def _placeESProducer(self,name,mod):
        self.__esproducers[name]=mod
    def _placeESPrefer(self,name,mod):
        self.__esprefers[name]=mod
    def _placeESSource(self,name,mod):
        self.__essources[name]=mod
    def _placePSet(self,name,mod):
        self.__psets[name]=mod
    def _placeVPSet(self,name,mod):
        self.__vpsets[name]=mod
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
    def _placeService(self,typeName,mod):
        self.__services[typeName]=mod
        self.__dict__[typeName]=mod
    def extend(self,other,items=()):
        """Look in other and find types which we can use"""
        seqs = dict()
        labelled = dict()
        for name in dir(other):
            item = getattr(other,name)
            if isinstance(item,_ModuleSequenceType):
                seqs[name]=item
                continue
            if isinstance(item,_Labelable):
                self.__setattr__(name,item)
                labelled[name]=item
                try:
                    item.label()
                except:
                    item.setLabel(name)
                continue
            if isinstance(item,_Unlabelable):
                self.add_(item)
        #now create a sequence which uses the newly made items
        for name in seqs.iterkeys():
            seq = seqs[name]
            newSeq = seq.copy()
            #
            self.__setattr__(name,newSeq)
    def _dumpConfigNamedList(self,items,typeName,indent):
        returnValue = ''
        for name,item in items:
            returnValue +=indent+typeName+' '+name+' = '+item.dumpConfig(indent,indent)
        return returnValue    
    def _dumpConfigUnnamedList(self,items,typeName,indent):
        returnValue = ''
        for name,item in items:
            returnValue +=indent+typeName+' = '+item.dumpConfig(indent,indent)
        return returnValue
    def _dumpConfigOptionallyNamedList(self,items,typeName,indent):
        returnValue = ''
        for name,item in items:
            if name == item.type_():
                name = ''
            else:
                name = ' '+name
            returnValue +=indent+typeName+name+' = '+item.dumpConfig(indent,indent)
        return returnValue
    def dumpConfig(self):
        """return a string containing the equivalent process defined using the configuration language"""
        config = "process "+self.__name+" = {\n"
        indent = "  "
        if self.source_():
            config += indent+"source = "+self.source_().dumpConfig(indent,indent)
        if self.looper_():
            config += indent+"looper = "+self.looper_().dumpConfig(indent,indent)
        config+=self._dumpConfigNamedList(self.producers_().iteritems(),
                                  'module',
                                  indent)
        config+=self._dumpConfigNamedList(self.filters_().iteritems(),
                                  'module',
                                  indent)
        config+=self._dumpConfigNamedList(self.analyzers_().iteritems(),
                                  'module',
                                  indent)
        config+=self._dumpConfigNamedList(self.outputModules_().iteritems(),
                                  'module',
                                  indent)
        config+=self._dumpConfigNamedList(self.sequences_().iteritems(),
                                  'sequence',
                                  indent)
        config+=self._dumpConfigNamedList(self.paths_().iteritems(),
                                  'path',
                                  indent)
        config+=self._dumpConfigNamedList(self.endpaths_().iteritems(),
                                  'endpath',
                                  indent)
        config+=self._dumpConfigUnnamedList(self.services_().iteritems(),
                                  'service',
                                  indent)
        config+=self._dumpConfigOptionallyNamedList(
            self.es_producers_().iteritems(),
            'es_module',
            indent)
        config+=self._dumpConfigOptionallyNamedList(
            self.es_sources_().iteritems(),
            'es_source',
            indent)
        config+=self._dumpConfigOptionallyNamedList(
            self.es_prefers_().iteritems(),
            'es_prefer',
            indent)
        for name,item in self.psets.iteritems():
            config +=indent+item.configTypeName()+' '+name+' = '+item.configValue(indent,indent)
        for name,item in self.vpsets.iteritems():
            config +=indent+'VPSet '+name+' = '+item.configValue(indent,indent)
        if self.schedule:
            pathNames = [p.label() for p in self.schedule]
            config +=indent+'schedule = {'+','.join(pathNames)+'}\n'
            
#        config+=self._dumpConfigNamedList(self.vpsets.iteritems(),
#                                  'VPSet',
#                                  indent)
        config += "}\n"
        return config

class FileInPath(_SimpleParameterTypeBase):
    def __init__(self,value):
        super(FileInPath,self).__init__(value)
    @staticmethod
    def _isValid(value):
        return True
    def configValue(self,indent,deltaIndent):
        return string.formatValueForConfig(self.value())
    @staticmethod
    def formatValueForConfig(value):
        return string.formatValueForConfig(value)
    @staticmethod
    def _valueFromString(value):
        return FileInPath(value)

class _Untracked(object):
    """Class type for 'untracked' to allow nice syntax"""
    __name__ = "untracked"
    @staticmethod
    def __call__(param):
        """used to set a 'param' parameter to be 'untracked'"""
        param.setIsTracked(False)
        return param
    def __getattr__(self,name):
        """A factory which allows syntax untracked.name(value) to construct an
        instance of 'name' class which is set to be untracked"""
        if name == "__bases__": raise AttributeError  # isclass uses __bases__ to recognize class objects 
        class Factory(object):
            def __init__(self,name):
                self.name = name
            def __call__(self,*value,**params):
                param = globals()[self.name](*value,**params)
                return _Untracked.__call__(param)
        return Factory(name)
#def untracked(param):
#    """used to set a 'param' parameter to be 'untracked'"""
#    param.setIsTracked(False)
#    return param
untracked = _Untracked()

class _Sequenceable(object):
    """Denotes an object which can be placed in a sequence"""
    def __mul__(self,rhs):
        return _SequenceOpAids(self,rhs)
    def __add__(self,rhs):
        return _SequenceOpFollows(self,rhs)
    def __invert__(self):
        return _SequenceNegation(self)
    def _clonesequence(self, lookuptable):
        try: 
            return lookuptable[id(self)]
        except:
            raise KeyError
        
class Service(_ConfigureComponent,_TypedParameterizable,_Unlabelable):
    def __init__(self,type,*arg,**kargs):
        super(Service,self).__init__(type,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeService(self.type_(),self)

class ESSource(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    def __init__(self,type,*arg,**kargs):
        super(ESSource,self).__init__(type,*arg,**kargs)
    def _placeImpl(self,name,proc):
        if name == '':
            name=self.type_()
        proc._placeESSource(name,self)

class ESProducer(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    def __init__(self,type,*arg,**kargs):
        super(ESProducer,self).__init__(type,*arg,**kargs)
    def _placeImpl(self,name,proc):
        if name == '':
            name=self.type_()
        proc._placeESProducer(name,self)

class ESPrefer(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    def __init__(self,type,*arg,**kargs):
        super(ESPrefer,self).__init__(type,*arg,**kargs)
    def _placeImpl(self,name,proc):
        if name == '':
            name=self.type_()
        proc._placeESPrefer(name,self)

class _Module(_ConfigureComponent,_TypedParameterizable,_Labelable,_Sequenceable):
    """base class for classes which denote framework event based 'modules'"""
    def __init__(self,type,*arg,**kargs):
        super(_Module,self).__init__(type,*arg,**kargs)

class EDProducer(_Module):
    def __init__(self,type,*arg,**kargs):
        super(EDProducer,self).__init__(type,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeProducer(name,self)
    pass

class EDFilter(_Module):
    def __init__(self,type,*arg,**kargs):
        super(EDFilter,self).__init__(type,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeFilter(name,self)
    pass

class EDAnalyzer(_Module):
    def __init__(self,type,*arg,**kargs):
        super(EDAnalyzer,self).__init__(type,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeAnalyzer(name,self)
    pass

class OutputModule(_Module):
    def __init__(self,type,*arg,**kargs):
        super(OutputModule,self).__init__(type,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeOutputModule(name,self)
    pass

class Source(_ConfigureComponent,_TypedParameterizable):
    def __init__(self,type,*arg,**kargs):
        super(Source,self).__init__(type,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeSource(name,self)

class Looper(_ConfigureComponent,_TypedParameterizable):
    def __init__(self,type,*arg,**kargs):
        super(Looper,self).__init__(type,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeLooper(name,self)

class _ModuleSequenceType(_ConfigureComponent, _Labelable):
    """Base class for classes which define a sequence of modules"""
    def __init__(self,first):
        self._seq = first
    def _place(self,name,proc):
        self._placeImpl(name,proc)
    def __imul__(self,rhs):
        self._seq = _SequenceOpAids(self._seq,rhs)
        return self
    def __iadd__(self,rhs):
        self._seq = _SequenceOpFollows(self._seq,rhs)
        return self
    def __str__(self):
        return str(self._seq)
    def dumpConfig(self,indent,deltaIndent):
        return '{'+self._seq.dumpSequenceConfig()+'}\n'
    def copy(self):
        returnValue =_ModuleSequenceType.__new__(type(self))
        returnValue.__init__(self._seq)
        return returnValue
    def _clonesequence(self, lookuptable):
        return type(self)(self._seq._clonesequence(lookuptable))
    #def replace(self,old,new):
    #"""Find all instances of old and replace with new"""
    #def insertAfter(self,which,new):
    #"""new will depend on which but nothing after which will depend on new"""
    #((a*b)*c)  >> insertAfter(b,N) >> ((a*b)*(N+c))
    #def insertBefore(self,which,new):
    #"""new will be independent of which"""
    #((a*b)*c) >> insertBefore(b,N) >> ((a*(N+b))*c)
    #def __contains__(self,item):
    #"""returns whether or not 'item' is in the sequence"""
    #def modules_(self):
    def _findDependencies(self,knownDeps,presentDeps):
        self._seq._findDependencies(knownDeps,presentDeps)
    def moduleDependencies(self):
        deps = dict()
        self._findDependencies(deps,set())
        return deps

class _SequenceOpAids(_Sequenceable):
    """Used in the expression tree for a sequence as a stand in for the ',' operator"""
    def __init__(self, left, right):
        self.__left = left
        self.__right = right
    def __str__(self):
        return '('+str(self.__left)+'*'+str(self.__right) +')'
    def dumpSequenceConfig(self):
        return '('+self.__left.dumpSequenceConfig()+','+self.__right.dumpSequenceConfig()+')'
    def _findDependencies(self,knownDeps,presentDeps):
        #do left first and then right since right depends on left
        self.__left._findDependencies(knownDeps,presentDeps)
        self.__right._findDependencies(knownDeps,presentDeps)
    def _clonesequence(self, lookuptable):
        return type(self)(self.__left._clonesequence(lookuptable),self.__right._clonesequence(lookuptable))


class _SequenceNegation(_Sequenceable):
    """Used in the expression tree for a sequence as a stand in for the '!' operator"""
    def __init__(self, operand):
        self.__operand = operand
    def __str__(self):
        return '!%s' %self.__operand
    def dumpSequenceConfig(self):
        return '!%s' %self.__operand.dumpSequenceConfig()
    def _findDependencies(self,knownDeps, presentDeps):
        self.__operand._findDependencies(knownDeps, presentDeps)

class _SequenceOpFollows(_Sequenceable):
    """Used in the expression tree for a sequence as a stand in for the '&' operator"""
    def __init__(self, left, right):
        self.__left = left
        self.__right = right
    def __str__(self):
        return '('+str(self.__left)+'+'+str(self.__right) +')'
    def dumpSequenceConfig(self):
        return '('+self.__left.dumpSequenceConfig()+'&'+self.__right.dumpSequenceConfig()+')'
    def _findDependencies(self,knownDeps,presentDeps):
        oldDepsL = presentDeps.copy()
        oldDepsR = presentDeps.copy()
        self.__left._findDependencies(knownDeps,oldDepsL)
        self.__right._findDependencies(knownDeps,oldDepsR)
        end = len(presentDeps)
        presentDeps.update(oldDepsL)
        presentDeps.update(oldDepsR)
    def _clonesequence(self, lookuptable):
        return type(self)(self.__left._clonesequence(lookuptable),self.__right._clonesequence(lookuptable))

class Path(_ModuleSequenceType):
    def __init__(self,first):
        super(Path,self).__init__(first)
    def _placeImpl(self,name,proc):
        proc._placePath(name,self)

class EndPath(_ModuleSequenceType):
    def __init__(self,first):
        super(EndPath,self).__init__(first)
    def _placeImpl(self,name,proc):
        proc._placeEndPath(name,self)

class Sequence(_ModuleSequenceType,_Sequenceable):
    def __init__(self,first):
        super(Sequence,self).__init__(first)
    def _placeImpl(self,name,proc):
        proc._placeSequence(name,self)

class Schedule(_ValidatingListBase,_ConfigureComponent,_Unlabelable):
    def __init__(self,*arg,**argv):
        super(Schedule,self).__init__(*arg,**argv)
    @staticmethod
    def _itemIsValid(item):
        return isinstance(item,Path) or isinstance(item,EndPath)
    def copy(self):
        import copy
        return copy.copy(self)
    def _place(self,label,process):
        process.setSchedule_(self)
        
def include(fileName):
    """Parse a configuration file language file and return a 'module like' object"""
    from FWCore.ParameterSet.parseConfig import importConfig
    return importConfig(fileName)

if __name__=="__main__":
    import unittest
    class TestModuleCommand(unittest.TestCase):
        def setUp(self):
            """Nothing to do """
            print 'testing'
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
        def testint32(self):
            i = int32(1)
            self.assertEqual(i.value(),1)
            self.assertRaises(ValueError,int32,"i")
            i = int32._valueFromString("0xA")
            self.assertEqual(i.value(),10)
        def testuint32(self):
            i = uint32(1)
            self.assertEqual(i.value(),1)
            i = uint32(0)
            self.assertEqual(i.value(),0)
            self.assertRaises(ValueError,uint32,"i")
            self.assertRaises(ValueError,uint32,-1)
            i = uint32._valueFromString("0xA")
            self.assertEqual(i.value(),10)
        def testvint32(self):
            v = vint32()
            self.assertEqual(len(v),0)
            v.append(1)
            self.assertEqual(len(v),1)
            self.assertEqual(v[0],1)
            v.append(2)
            v.insert(1,3)
            self.assertEqual(v[1],3)
            v[1]=4
            self.assertEqual(v[1],4)
            v[1:1]=[5]
            self.assertEqual(len(v),4)
            self.assertEqual([1,5,4,2],list(v))
            self.assertRaises(TypeError,v.append,('blah'))
        def testString(self):
            s=string('this is a test')
            self.assertEqual(s.value(),'this is a test')
            s=string('\0')
            self.assertEqual(s.value(),'\0')
            self.assertEqual(s.configValue('',''),"'\\0'")
        def testUntracked(self):
            p=untracked(int32(1))
            self.assertRaises(TypeError,untracked,(1),{})
            self.failIf(p.isTracked())
            p=untracked.int32(1)
            self.assertRaises(TypeError,untracked,(1),{})
            self.failIf(p.isTracked())
            p=untracked.vint32(1,5,3)
            self.assertRaises(TypeError,untracked,(1,5,3),{})
            self.failIf(p.isTracked())
            p = untracked.PSet(b=int32(1))
            self.failIf(p.isTracked())
            self.assertEqual(p.b.value(),1)

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
            print p.es_sources_().keys()
            self.assert_('geom' in p.es_sources_())
            p.add_(ESSource("ConfigDB"))
            self.assert_('ConfigDB' in p.es_sources_())

        def testProcessExtend(self):
            class FromArg(object):
                def __init__(self,*arg,**args):
                    for name in args.iterkeys():
                        self.__dict__[name]=args[name]
            
            a=EDAnalyzer("MyAnalyzer")
            d = FromArg(
                    a=a,
                    b=Service("Full"),
                    c=Path(a),
                    d=Sequence(a)
                )
            p = Process("Test")
            p.extend(d)
            self.assertEqual(p.a.type_(),"MyAnalyzer")
            self.assertRaises(AttributeError,getattr,p,'b')
            self.assertEqual(p.Full.type_(),"Full")
            self.assertEqual(str(p.c),'a')
            self.assertEqual(str(p.d),'a')
        def testProcessDumpConfig(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.paths = Path(p.a)
            p.dumpConfig()
        def testEDAnalyzer(self):
            empty = EDAnalyzer("Empty")
            withParam = EDAnalyzer("Parameterized",foo=untracked(int32(1)), bar = untracked(string("it")))
            self.assertEqual(withParam.foo.value(), 1)
            self.assertEqual(withParam.bar.value(), "it")
            aCopy = withParam.copy()
            self.assertEqual(aCopy.foo.value(), 1)
            self.assertEqual(aCopy.bar.value(), "it")
            
        def testService(self):
            empty = Service("Empty")
            withParam = Service("Parameterized",foo=untracked(int32(1)), bar = untracked(string("it")))
            self.assertEqual(withParam.foo.value(), 1)
            self.assertEqual(withParam.bar.value(), "it")
            
        def testSequence(self):
            p = Process('test')
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.s = Sequence(p.a*p.b)
            self.assertEqual(str(p.s),'(a*b)')
            self.assertEqual(p.s.label(),'s')
            path = Path(p.c+p.s)
            self.assertEqual(str(path),'(c+(a*b))')

        def testPath(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            path = Path(p.a)
            path *= p.b
            path += p.c
            self.assertEqual(str(path),'((a*b)+c)')
            path = Path(p.a*p.b+p.c)
            self.assertEqual(str(path),'((a*b)+c)')
#            path = Path(p.a)*p.b+p.c #This leads to problems with sequences
#            self.assertEqual(str(path),'((a*b)+c)')
            path = Path(p.a+ p.b*p.c)
            self.assertEqual(str(path),'(a+(b*c))')
            path = Path(p.a*(p.b+p.c))
            self.assertEqual(str(path),'(a*(b+c))')
            path = Path(p.a*(p.b+~p.c)) 
            self.assertEqual(str(path),'(a*(b+!c))')

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
            self.assertEqual(str(path),str(path._clonesequence(lookuptable)))
            lookuptable = p._cloneToObjectDict
            self.assertEqual(str(path),str(path._clonesequence(lookuptable)))
            self.assertEqual(str(path),str(p.path))
            
        def testSchedule(self):
            p = Process("test")
            p.a = EDAnalyzer("MyAnalyzer")
            p.b = EDAnalyzer("YourAnalyzer")
            p.c = EDAnalyzer("OurAnalyzer")
            p.path1 = Path(p.a)
            p.path2 = Path(p.b)
            
            s = Schedule(p.path1,p.path2)
            self.assertEqual(s[0],p.path1)
            self.assertEqual(s[1],p.path2)
            p.schedule = s
        def testExamples(self):
            p = Process("Test")
            p.source = Source("PoolSource",fileNames = untracked(string("file:reco.root")))
            p.foos = EDProducer("FooProducer")
            p.bars = EDProducer("BarProducer", foos=InputTag("foos"))
            p.out = OutputModule("PoolOutputModule",fileName=untracked(string("file:foos.root")))
            p.p = Path(p.foos*p.bars)
            p.e = EndPath(p.out)
            p.add_(Service("MessageLogger"))
        def testFindDependencies(self):
            p = Process("test")
            p.a = EDProducer("MyProd")
            p.b = EDProducer("YourProd")
            p.c = EDProducer("OurProd")
            path = Path(p.a)
            path *= p.b
            path += p.c
            print 'denpendencies'
            deps= path.moduleDependencies()
            self.assertEqual(deps['a'],set())
            self.assertEqual(deps['b'],set(['a']))
            self.assertEqual(deps['c'],set())
            
            path *=p.a
            print str(path)
            self.assertRaises(RuntimeError,path.moduleDependencies)
            path = Path(p.a*(p.b+p.c))
            deps = path.moduleDependencies()
            self.assertEqual(deps['a'],set())
            self.assertEqual(deps['b'],set(['a']))
            self.assertEqual(deps['c'],set(['a']))
            #deps= path.moduleDependencies()
            #print deps['a']
        def testFixedKeysDict(self):
            import operator
            d = FixedKeysDict({'a':1, 'b':[3]})
            self.assertEqual(d['a'],1)
            self.assertEqual(d['b'],[3])
            self.assertRaises(AttributeError,operator.setitem,*(d,'a',2))
            d['b'].append(2)
            self.assertEqual(d['b'],[3,2])
        def testSortedKeysDict(self):
            sd = SortedKeysDict()
            sd['a']=1
            sd['b']=2
            sd['c']=3
            sd['d']=4
            count =1
            for key in sd.iterkeys():
                self.assertEqual(count,sd[key])
                count +=1
            sd2 = SortedKeysDict(sd)
            count =1
            for key in sd2.iterkeys():
                self.assertEqual(count,sd2[key])
                count +=1
            sd3 = SortedKeysDict([('a',1),('b',2),('c',3),('d',4)])
            count =1
            for key in sd3.iterkeys():
                self.assertEqual(count,sd3[key])
                count +=1
            self.assertEqual(count-1,len(sd3))
            sd3 = SortedKeysDict(a=1,b=2,c=3,d=4)
            count =1
            for key in sd3.iterkeys():
                count +=1
            self.assertEqual(count-1,len(sd3))
        

        def testSortedAndFixedKeysDict(self):
            import operator
            sd = SortedAndFixedKeysDict({'a':1, 'b':[3]})
            self.assertEqual(sd['a'],1)
            self.assertEqual(sd['b'],[3])
            self.assertRaises(AttributeError,operator.setitem,*(sd,'a',2))
                               
    unittest.main()
