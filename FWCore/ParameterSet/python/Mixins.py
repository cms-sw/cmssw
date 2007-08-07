
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
    def insertContentsInto(self, parameterSet):
        for name in self.parameterNames_():
            param = getattr(self,name)
            param.insertInto(parameterSet, name)


class _TypedParameterizable(_Parameterizable):
    """Base class for classes which are Parameterizable and have a 'type' assigned"""
    def __init__(self,type_,*arg,**kargs):
        self.__dict__['_TypedParameterizable__type'] = type_
        #the 'type' is also placed in the 'arg' list and we need to remove it
        if 'type_' not in kargs:
            arg = arg[1:]
        else:
            del args['type_']
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
    def nameInProcessDesc_(self, myname):
        return myname;
    def insertInto(self, parameterSet, myname):
        newpset = parameterSet.newPSet()
        newpset.addString(True, "@module_label", self.nameInProcessDesc_(myname))
        newpset.addString(True, "@module_type", self.type_())
        self.insertContentsInto(newpset)
        parameterSet.addPSet(True, myname, newpset)



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
