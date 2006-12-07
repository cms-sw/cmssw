#!/usr/bin/env python

# helper classes for sorted and fixed dicts

class SortedKeysDict(dict):
    """a dict preserving order of keys"""
    # specialised __repr__ missing.
    def __init__(self):
        dict.__init__(self)
        self.list = list()

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
        new = sortedKeysDict.__new__(cls)
        sortedKeysDict.__init__(new, *args, **kw)
        return new
    def __init__(self, *args, **kw):
        pass
    def __repr__(self):
        return "SortedAndFixedKeysDict(%s)" % sortedKeysDict.__repr__(self)

    
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
    """Look in side the module and find the Processes it contains"""
    process = None
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
        self.__dict__['_Process__analyzers'] = {}
        self.__dict__['_Process__outputmodules'] = {}
        self.__dict__['_Process__paths'] = SortedKeysDict()    # have to keep the order
        self.__dict__['_Process__endpaths'] = SortedKeysDict() # of definition
        self.__dict__['_Process__sequences'] = {}
        self.__dict__['_Process__services'] = {}
        self.__dict__['_Process__essources'] = {}
        self.__dict__['_Process__esproducers'] = {}
        self.__dict__['_Process__psets']={}
        self.__dict__['_Process__vpsets']={}
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
        if not isinstance(value,_Labelable) and not isinstance(value,Source) and not isinstance(value,Looper):
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
        #NOTE: NEED TO ALSO RECORD THE ORDER THESE WERE ADDED SINCE
        # THAT WILL BE THE DEFAULT ORDER OF RUNNING
        self.__paths[name]=mod
    def _placeEndPath(self,name,mod):
        self.__endpaths[name]=mod
    def _placeSequence(self,name,mod):
        self.__sequences[name]=mod
    def _placeESProducer(self,name,mod):
        self.__esproducers[name]=mod
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
    def useFrom(self,other,items=()):
        """Look in the __dict__ of other and find types which we can use"""
        seqs = dict()
        labelled = dict()
        for name in other.__dict__.iterkeys():
            item = other.__dict__[name]
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
        for name,item in self.psets.iteritems():
            config +=indent+item.configTypeName()+' '+name+' = '+item.configValue(indent,indent)
        for name,item in self.vpsets.iteritems():
            config +=indent+'VPSet '+name+' = '+item.configValue(indent,indent)
#        config+=self._dumpConfigNamedList(self.vpsets.iteritems(),
#                                  'VPSet',
#                                  indent)
        config += "}\n"
        return config
    
class _ConfigureComponent(object):
    """Denotes a class that can be used by the Processes class"""
    pass
        
class _Parameterizable(object):
    """Base class for classes which allow addition of _ParameterTypeBase data"""
    def __init__(self,*arg,**kargs):
        """The named arguments are the 'parameters' which are added as 'python attributes' to the object"""
        if len(arg) != 0:
            raise ValueError("unnamed arguments are not allowed. Please use the syntax 'name = value' when assigning arguments.")
        self.__dict__['_Parameterizable__parameterNames'] = []
        self.__setParameters(kargs)
    def parameterNames_(self):
        """Returns the name of the parameters"""
        return self.__parameterNames[:]
    def __setParameters(self,parameters):
        for name,value in parameters.iteritems():
            if not isinstance(value,_ParameterTypeBase):
                raise TypeError
            self.__dict__[name]=value
            self.__parameterNames.append(name)
    def __setattr__(self,name,value):
        #since labels are not supposed to have underscores at the beginning
        # I will assume that if we have such then we are setting an internal variable
        if name[0]=='_':
            super(_Parameterizable,self).__setattr__(name,value)
        if not name in self.__dict__:
            if not isinstance(value,_ParameterTypeBase):
                raise TypeError
            self.__dict__[name]=value
            self.__parameterNames.append(name)
        param = self.__dict__[name]
        if not isinstance(param,_ParameterTypeBase):
            self.__dict__[name]=value
        else:
            if isinstance(value,_ParameterTypeBase):
                self.__dict__[name] =value
            else:
                param.setValue(value)
    def __delattr__(self,name):
        super(_Parameterizable,self).__delattr__(name)
        self.__parameterNames.remove(name)

class _TypedParameterizable(_Parameterizable):
    """Base class for classes which are Parameterizable and have a 'type' assigned"""
    def __init__(self,type,*arg,**kargs):
        self.__dict__['_TypedParameterizable__type'] = type
        #the 'type' is also placed in the 'arg' list and we need to remove it
        if 'type' not in kargs:
            arg = arg[1:]
        else:
            del args['type']
        super(_TypedParameterizable,self).__init__(*arg,**kargs)
    def _place(self,name,proc):
        self._placeImpl(name,proc)
    def type_(self):
        """returns the type of the object, e.g. 'FooProducer'"""
        return self.__type
    def copy(self):
        import copy
        returnValue =_TypedParameterizable.__new__(type(self))
        params = dict()
        for name in self.parameterNames_():
               params[name]=copy.deepcopy(self.__dict__[name])
        args = list()
        if len(params) == 0:
            args.append(None)
        returnValue.__init__(self.__type,*args,
                             **params)
        return returnValue
    @staticmethod
    def __findDefaultsFor(label,type):
        #This routine is no longer used, but I might revive it in the future
        import sys
        import glob
        choices = list()
        for d in sys.path:
            choices.extend(glob.glob(d+'/*/*/'+label+'.py'))
        if not choices:
            return None
        #now see if any of them have what we want
        #the use of __import__ is taken from an example
        # from the www.python.org documentation on __import__
        for c in choices:
            #print " found file "+c
            name='.'.join(c[:-3].split('/')[-3:])
            #name = c[:-3].replace('/','.')
            mod = __import__(name)
            components = name.split('.')
            for comp in components[1:]:
                mod = getattr(mod,comp)
            if hasattr(mod,label):
                default = getattr(mod,label)
                if isinstance(default,_TypedParameterizable):
                    if(default.type_() == type):
                        params = dict()
                        for name in default.parameterNames_():
                            params[name] = getattr(default,name)
                        return params
        return None
    
    def dumpConfig(self,indent='',deltaIndent=''):
        config = self.__type +' { \n'
        for name in self.parameterNames_():
            param = self.__dict__[name]
            config+=indent+deltaIndent+param.configTypeName()+' '+name+' = '+param.configValue(indent+deltaIndent,deltaIndent)+'\n'
        config += indent+'}\n'
        return config
        
class _Labelable(object):
    """A 'mixin' used to denote that the class can be paired with a label (e.g. an EDProducer)"""
    def setLabel(self,label):
        self.__label = label
    def label(self):
        return self.__label
    def __str__(self):
        #this is probably a bad idea
        # I added this so that when we ask a path to print
        # we will see the label that has been assigned
        return str(self.__label)
    def dumpSequenceConfig(self):
        return str(self.__label)
    def _findDependencies(self,knownDeps,presentDeps):
        #print 'in labelled'
        myDeps=knownDeps.get(self.label(),None)
        if myDeps!=None:
            if presentDeps != myDeps:
                raise RuntimeError("the module "+self.label()+" has two dependencies \n"
                                   +str(presentDeps)+"\n"
                                   +str(myDeps)+"\n"
                                   +"Please modify sequences to rectify this inconsistency")
        else:
            myDeps=set(presentDeps)
            knownDeps[self.label()]=myDeps
        presentDeps.add(self.label())


class _Unlabelable(object):
    """A 'mixin' used to denote that the class can be used without a label (e.g. a Service)"""
    pass

class _ParameterTypeBase(object):
    """base class for classes which are used as the 'parameters' for a ParameterSet"""
    def __init__(self):
        self.__isTracked = True
    def configTypeName(self):
        if self.isTracked():            
            return type(self).__name__
        return 'untracked '+type(self).__name__
    def isTracked(self):
        return self.__isTracked
    def setIsTracked(self,trackness):
        self.__isTracked = trackness

class _SimpleParameterTypeBase(_ParameterTypeBase):
    """base class for parameter classes which only hold a single value"""
    def __init__(self,value):
        super(_SimpleParameterTypeBase,self).__init__()
        self._value = value
        if not self._isValid(value):
            raise ValueError(str(value)+" is not a valid "+str(type(self)))        
    def value(self):
        return self._value
    def setValue(self,value):
        if not self._isValid(value):
            raise ValueError(str(value)+" is not a valid "+str(type(self)))        
        self._value = value
    def configValue(self,indent,deltaIndent):
        return str(self._value)

class int32(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return isinstance(value,int)
    @staticmethod
    def _valueFromString(value):
        if len(value) >1 and '0x' == value[:2]:
            return int32(int(value,16))
        return int32(int(value))

class uint32(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return ((isinstance(value,int) and value > 0) or
                (isinstance(value,long) and value > 0) and value <= 0xFFFFFFFF)
    @staticmethod
    def _valueFromString(value):
        if len(value) >1 and '0x' == value[:2]:
            return uint32(long(value,16))
        return uint32(long(value))

class int64(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return isinstance(value,int) or (
            isinstance(value,long) and
            (-0x7FFFFFFFFFFFFFFF < value <= 0x7FFFFFFFFFFFFFFF) )
    @staticmethod
    def _valueFromString(value):
        if len(value) >1 and '0x' == value[:2]:
            return uint32(long(value,16))
        return int64(long(value))

class uint64(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return ((isinstance(value,int) and value > 0) or
                (ininstance(value,long) and value > 0) and value <= 0xFFFFFFFFFFFFFFFF)
    @staticmethod
    def _valueFromString(value):
        if len(value) >1 and '0x' == value[:2]:
            return uint32(long(value,16))
        return uint64(long(value))

class double(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return True
    @staticmethod
    def _valueFromString(value):
        return double(float(value))

class bool(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return (isinstance(value,type(False)) or isinstance(value(type(True))))
    @staticmethod
    def _valueFromString(value):
        if (value.lower() == 'true' or
            value.lower() == 't' or
            value.lower() == 'on' or
            value.lower() == 'yes' or
            value.lower() == '1'):
            return bool(True)
        if (value.lower() == 'false' or
            value.lower() == 'f' or
            value.lower() == 'off' or
            value.lower() == 'no' or
            value.lower() == '0' ):
            return bool(False)
        raise RuntimeError('can not make bool from string '+value)
        

class string(_SimpleParameterTypeBase):
    def __init__(self,value):
        super(string,self).__init__(value)
    @staticmethod
    def _isValid(value):
        return isinstance(value,type(''))
    def configValue(self,indent,deltaIndent):
        return self.formatValueForConfig(self.value())
    @staticmethod
    def formatValueForConfig(value):
        if "'" in value:
            return '"'+value+'"'
        return "'"+value+"'"
    @staticmethod
    def _valueFromString(value):
        return string(value)

class InputTag(_ParameterTypeBase):
    def __init__(self,moduleLabel,productInstanceLabel=''):
        super(InputTag,self).__init__()
        self.__moduleLabel = moduleLabel
        self.__productInstance = productInstanceLabel
    def getModuleLabel(self):
        return self.__moduleLabel
    def setModuleLabel(self,label):
        self.__moduleLabel = label
    moduleLabel = property(getModuleLabel,setModuleLabel,"module label for the product")
    def getProductInstanceLabel(self):
        return self.__productInstance
    def setProductInstanceLabel(self,label):
        self.__productInstance = label
    productInstanceLabel = property(getProductInstanceLabel,setProductInstanceLabel,"product instance label for the product")
    def configValue(self,indent,deltaIndent):
        return self.__moduleLabel+':'+self.__productInstance
    @staticmethod
    def _isValid(value):
        return True
    def __cmp__(self,other):
        v = self.__moduleLabel <> other.__moduleLabel
        if not v:
            return self.__productInstance <> other.__productInstance
        return v
    @staticmethod
    def formatValueForConfig(value):
        return value.configValue('','')

class FileInPath(_SimpleParameterTypeBase):
    def __init__(self,value):
        super(InputTag,self).__init__(value)
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
        return string(value)

class PSet(_ParameterTypeBase,_Parameterizable,_ConfigureComponent,_Labelable):
    def __init__(self,*arg,**args):
        #need to call the inits separately
        _ParameterTypeBase.__init__(self)
        _Parameterizable.__init__(self,*arg,**args)
    def value(self):
        return self
    @staticmethod
    def _isValid(value):
        return True
    def configValue(self,indent='',deltaIndent=''):
        config = '{ \n'
        for name in self.parameterNames_():
            param = getattr(self,name)
            config+=indent+deltaIndent+param.configTypeName()+' '+name+' = '+param.configValue(indent+deltaIndent,deltaIndent)+'\n'
        config += indent+'}\n'
        return config
    def copy(self):
        import copy
        return copy.copy(self)
    def _place(self,name,proc):
        proc._placePSet(name,self)
    def __str__(self):
        return object.__str__(self)

class _ValidatingListBase(list):
    """Base class for a list which enforces that its entries pass a 'validity' test"""
    def __init__(self,*arg,**args):        
        super(_ValidatingListBase,self).__init__(arg)
        if not self._isValid(iter(self)):
            raise TypeError("wrong types added to "+str(type(self)))
    def __setitem__(self,key,value):
        if isinstance(key,slice):
            if not self._isValid(value):
                raise TypeError("wrong type being inserted into this container")
        else:
            if not self._itemIsValid(value):
                raise TypeError("can not insert the type "+str(type(value))+" in this container")
        super(_ValidatingListBase,self).__setitem__(key,value)
    def _isValid(self,seq):
        for item in seq:
            if not self._itemIsValid(item):
                return False
        return True
    def append(self,x):
        if not self._itemIsValid(x):
            raise TypeError("wrong type being appended to this container")
        super(_ValidatingListBase,self).append(x)
    def extend(self,x):
        if not self._isValid(x):
            raise TypeError("wrong type being extended to this container")
        super(_ValidatingListBase,self).extend(x)
    def insert(self,i,x):
        if not self._itemIsValid(x):
            raise TypeError("wrong type being inserted to this container")
        super(_ValidatingListBase,self).insert(i,x)
    def value(self):
        return list(self)
    def setValue(self,v):
        self[:] = []
        self.extend(v)
    def configValue(self,indent,deltaIndent):
        config = '{\n'
        first = True
        for value in iter(self):
            config +=indent+deltaIndent
            if not first:
                config+=', '
            config+=  self.configValueForItem(value,indent,deltaIndent)+'\n'
            first = False
        config += indent+'}\n'
        return config
    def configValueForItem(self,item,indent,deltaIndent):
        return str(item)
    @staticmethod
    def _itemsFromStrings(strings,converter):
        return (converter(x).value() for x in strings)
        

class vint32(_ValidatingListBase,_ParameterTypeBase):
    def __init__(self,*arg,**args):
        _ParameterTypeBase.__init__(self)
        super(vint32,self).__init__(*arg,**args)
        
    @staticmethod
    def _itemIsValid(item):
        return int32._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vint32(*_ValidatingListBase._itemsFromStrings(value,int32._valueFromString))

class vuint32(_ValidatingListBase,_ParameterTypeBase):
    def __init__(self,*arg,**args):
        _ParameterTypeBase.__init__(self)
        super(vuint32,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return uint32._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vuint32(*_ValidatingListBase._itemsFromStrings(value,uint32._valueFromString))
    
class vint64(_ValidatingListBase,_ParameterTypeBase):
    def __init__(self,*arg,**args):
        _ParameterTypeBase.__init__(self)
        super(vint64,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return int64._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vint64(*_ValidatingListBase._itemsFromStrings(value,int64._valueFromString))

class vuint64(_ValidatingListBase,_ParameterTypeBase):
    def __init__(self,*arg,**args):
        _ParameterTypeBase.__init__(self)
        super(vuint64,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return uint64._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vuint64(*_ValidatingListBase._itemsFromStrings(value,vuint64._valueFromString))
    
class vdouble(_ValidatingListBase,_ParameterTypeBase):
    def __init__(self,*arg,**args):
        _ParameterTypeBase.__init__(self)
        super(vdouble,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return double._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vdouble(*_ValidatingListBase._itemsFromStrings(value,double._valueFromString))

class vbool(_ValidatingListBase,_ParameterTypeBase):
    def __init__(self,*arg,**args):
        _ParameterTypeBase.__init__(self)
        super(vbool,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return bool._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vbool(*_ValidatingListBase._itemsFromStrings(value,bool._valueFromString))

class vstring(_ValidatingListBase,_ParameterTypeBase):
    def __init__(self,*arg,**args):
        _ParameterTypeBase.__init__(self)
        super(vstring,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return string._isValid(item)
    def configValueForItem(self,item,indent,deltaIndent):
        return string.formatValueForConfig(item)
    @staticmethod
    def _valueFromString(value):
        return vstring(*_ValidatingListBase._itemsFromStrings(value,string._valueFromString))

class VInputTag(_ValidatingListBase,_ParameterTypeBase):
    def __init__(self,*arg,**args):
        _ParameterTypeBase.__init__(self)
        super(VInputTag,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return InputTag._isValid(item)
    def configValueForItem(self,item,indent,deltaIndent):
        return InputTag.formatValueForConfig(item)
    @staticmethod
    def _valueFromString(value):
        return VInputTag(*_ValidatingListBase._itemsFromStrings(value,InputTag._valueFromString))

class VPSet(_ValidatingListBase,_ParameterTypeBase,_ConfigureComponent,_Labelable):
    def __init__(self,*arg,**args):
        _ParameterTypeBase.__init__(self)
        super(VPSet,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return PSet._isValid(item)
    def configValueForItem(self,item,indent,deltaIndent):
        return PSet.configValue(item,indent+deltaIndent,deltaIndent)
    def copy(self):
        import copy
        return copy.copy(self)
    def _place(self,name,proc):
        proc._placeVPSet(name,self)

def untracked(param):
    """used to set a 'param' parameter to be 'untracked'"""
    param.setIsTracked(False)
    return param

class _Sequenceable(object):
    """Denotes an object which can be placed in a sequence"""
    def __mul__(self,rhs):
        return _SequenceOpAids(self,rhs)
    def __add__(self,rhs):
        return _SequenceOpFollows(self,rhs)

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
    def __init__(self,left,right):
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

class _SequenceOpFollows(_Sequenceable):
    """Used in the expression tree for a sequence as a stand in for the '&' operator"""
    def __init__(self,left,right):
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
            
        def testUntracked(self):
            p=untracked(int32(1))
            self.assertRaises(TypeError,untracked,(1),{})
            self.failIf(p.isTracked())
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

        def testProcessUseFrom(self):
            class FromArg(object):
                def __init__(self,*arg,**args):
                    for name in args.iterkeys():
                        self.__dict__[name]=args[name]
            
            a=EDAnalyzer("MyAnalyzer")
            d = FromArg(
                    a=a,
                    b=Service("Full"),
                    c=Path(a)
                )
            p = Process("Test")
            p.useFrom(d)
            self.assertEqual(p.a.type_(),"MyAnalyzer")
            self.assertRaises(AttributeError,getattr,p,'b')
            self.assertEqual(p.Full.type_(),"Full")
            self.assertEqual(str(p.c),'a')
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
                               
    unittest.main()
