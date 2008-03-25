
class _ConfigureComponent(object):
    """Denotes a class that can be used by the Processes class"""
    pass

class PrintOptions(object):
    def __init__(self):
        self.indent_= 0
        self.deltaIndent_ = 4
        self.isCfg = True
    def indentation(self):
        return ' '*self.indent_
    def indent(self):
        self.indent_ += self.deltaIndent_
    def unindent(self):
        self.indent_ -= self.deltaIndent_

class _ParameterTypeBase(object):
    """base class for classes which are used as the 'parameters' for a ParameterSet"""
    def __init__(self):
        self.__isTracked = True
    def configTypeName(self):
        if self.isTracked():
            return type(self).__name__
        return 'untracked '+type(self).__name__
    def pythonTypeName(self):
        if self.isTracked():
            return 'cms.'+type(self).__name__
        return 'cms.untracked.'+type(self).__name__
    def dumpPython(self, options=PrintOptions()):
        return self.pythonTypeName()+"("+self.pythonValue(options)+")"
    def __repr__(self):
        return self.dumpPython()
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
    def configValue(self, options=PrintOptions()):
        return str(self._value)
    def pythonValue(self, options=PrintOptions()):
        return self.configValue(options)

class UsingBlock(_SimpleParameterTypeBase):
    """For injection purposes, pretend this is a new parameter type
       then have a post process step which strips these out
    """
    def __init__(self,value):
        super(UsingBlock,self).__init__(value)
        self.isResolved = False
    @staticmethod
    def _isValid(value):
        return isinstance(value,str)
    def _valueFromString(value):
        """only used for cfg-parsing"""
        return string(value)
    def insertInto(self, parameterSet, myname):
        value = self.value()
        #  doesn't seem to handle \0 correctly
        #if value == '\0':
        #    value = ''
        parameterSet.addString(self.isTracked(), myname, value)
    def dumpPython(self, options):
        if options.isCfg:
            return "process."+self.value()
        else:
            return self.value()


class _Parameterizable(object):
    """Base class for classes which allow addition of _ParameterTypeBase data"""
    def __init__(self,*arg,**kargs):
        self.__dict__['_Parameterizable__parameterNames'] = []
        """The named arguments are the 'parameters' which are added as 'python attributes' to the object"""
        if len(arg) != 0:
            #raise ValueError("unnamed arguments are not allowed. Please use the syntax 'name = value' when assigning arguments.")
            for block in arg:
                if type(block).__name__ != "PSet":
                    raise ValueError("Only PSets can be passed as unnamed argument blocks.  This is a "+type(block).__name__)
                self.__setParameters(block.parameters())
        self.__setParameters(kargs)
        self._isModified = False
    def parameterNames_(self):
        """Returns the name of the parameters"""
        return self.__parameterNames[:]
    def isModified(self):
        if self._isModified:
            return True
        for name in self.parameterNames_():
            param = self.__dict__[name]
            if isinstance(param, _Parameterizable) and param.isModified():
                self._isModified = True
                return True
        return False
    def parameters(self):
        """Returns a dictionary of copies of the user-set parameters"""
        import copy
        result = dict()
        for name in self.parameterNames_():
               result[name]=copy.deepcopy(self.__dict__[name])
        return result
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
            # RICK test
            return
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
        self._isModified = True
    def __delattr__(self,name):
        super(_Parameterizable,self).__delattr__(name)
        self.__parameterNames.remove(name)
    def dumpPython(self, options=PrintOptions()):
        others = []
        usings = []
        for name in self.parameterNames_():
            param = self.__dict__[name]
            # we don't want minuses in names
            name2 = name.replace('-','_')
            options.indent()
            #_UsingNodes don't get assigned variables
            if name.startswith("using_"):
                usings.append(options.indentation()+param.dumpPython(options))
            else:
                others.append(options.indentation()+name2+' = '+param.dumpPython(options))
            options.unindent()
        # usings need to go first
        resultList = usings
        resultList.extend(others)
        return ',\n'.join(resultList)+'\n'
    def __repr__(self):
        return self.dumpPython()
    def insertContentsInto(self, parameterSet):
        for name in self.parameterNames_():
            param = getattr(self,name)
            param.insertInto(parameterSet, name)


class _TypedParameterizable(_Parameterizable):
    """Base class for classes which are Parameterizable and have a 'type' assigned"""
    def __init__(self,type_,*arg,**kargs):
        self.__dict__['_TypedParameterizable__type'] = type_
        #the 'type' is also placed in the 'arg' list and we need to remove it
        #if 'type_' not in kargs:
        #    arg = arg[1:]
        #else:
        #    del args['type_']
        arg = tuple([x for x in arg if x != None])
        super(_TypedParameterizable,self).__init__(*arg,**kargs)
    def _place(self,name,proc):
        self._placeImpl(name,proc)
    def type_(self):
        """returns the type of the object, e.g. 'FooProducer'"""
        return self.__type
    def copy(self):
        returnValue =_TypedParameterizable.__new__(type(self))
        params = self.parameters()
        args = list()
        if len(params) == 0:
            args.append(None)
        returnValue.__init__(self.__type,*args,
                             **params)
        returnValue._isModified = self._isModified
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
    
    def dumpConfig(self, options=PrintOptions()):
        config = self.__type +' { \n'
        for name in self.parameterNames_():
            param = self.__dict__[name]
            options.indent()
            config+=options.indentation()+param.configTypeName()+' '+name+' = '+param.configValue(options)+'\n'
            options.unindent()
        config += options.indentation()+'}\n'
        return config

    def dumpPython(self, options=PrintOptions()):
        result = "cms."+str(type(self).__name__)+"(\""+self.type_()+"\""
        nparam = len(self.parameters())
        if nparam == 0:
            result += ")\n"
        elif nparam < 256:
            result += ",\n"+_Parameterizable.dumpPython(self,options)+options.indentation() + ")\n"
        else:
            # too big.  Need to dump externally
            label = ""
            try:
               label = self.label()
            except:
               label = "FIX-THIS"
            result += ")\n" + self.dumpPythonAttributes(label, options)
        return result

    def dumpPythonAttributes(self, myname, options):
        """ dumps the object with all attributes declared after the constructor"""
        result = ""
        for name in self.parameterNames_():
            param = self.__dict__[name]
            result += options.indentation() + myname + "." + name + " = " + param.dumpPython(options) + "\n"
        return result

    def nameInProcessDesc_(self, myname):
        return myname;
    def moduleLabel_(self, myname):
        return myname
    def insertInto(self, parameterSet, myname):
        newpset = parameterSet.newPSet()
        newpset.addString(True, "@module_label", self.moduleLabel_(myname))
        newpset.addString(True, "@module_type", self.type_())
        self.insertContentsInto(newpset)
        parameterSet.addPSet(True, self.nameInProcessDesc_(myname), newpset)



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
    def dumpSequencePython(self):
        return 'process.'+str(self.__label)
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
    def fillNamesList(self, l, otherSequences):
        l.append(self.__label)


class _Unlabelable(object):
    """A 'mixin' used to denote that the class can be used without a label (e.g. a Service)"""
    pass

class _ValidatingListBase(list):
    """Base class for a list which enforces that its entries pass a 'validity' test"""
    def __init__(self,*arg,**args):        
        super(_ValidatingListBase,self).__init__(arg)
        if not self._isValid(iter(self)):
            raise TypeError("wrong types ("+','.join([str(type(value)) for value in iter(self)])+
                            ") added to "+str(type(self)))
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
    def __add__(self,rhs):
        if not self._isValid(rhs):
            raise TypeError("wrong type being added to this container")
        import copy
        value = copy.copy(self)
        value.extend(rhs)
        return value
    def insert(self,i,x):
        if not self._itemIsValid(x):
            raise TypeError("wrong type being inserted to this container")
        super(_ValidatingListBase,self).insert(i,x)

class _ValidatingParameterListBase(_ValidatingListBase,_ParameterTypeBase):
    def __init__(self,*arg,**args):
        _ParameterTypeBase.__init__(self)
        super(_ValidatingParameterListBase,self).__init__(*arg,**args)
    def value(self):
        return list(self)
    def setValue(self,v):
        self[:] = []
        self.extend(v)
    def configValue(self, options=PrintOptions()):
        config = '{\n'
        first = True
        for value in iter(self):
            options.indent()
            config += options.indentation()
            if not first:
                config+=', '
            config+=  self.configValueForItem(value, options)+'\n'
            first = False
            options.unindent()
        config += options.indentation()+'}\n'
        return config
    def configValueForItem(self,item, options):
        return str(item)
    def pythonValueForItem(self,item, options):
        return self.configValueForItem(item, options)
    def __repr__(self):
        return self.dumpPython()
    def dumpPython(self, options=PrintOptions()):
        result = self.pythonTypeName()+"("
        if len(self)<255:
            result+=', '.join((self.pythonValueForItem(v,options) for v in iter(self)))
        else:
            result = '('
            start = 0
            l=list()
            while(start < len(self)):
                v=self.pythonTypeName()+'('
                v+=', '.join((self.pythonValueForItem(v,options) for v in self[start:start+255]))
                v+=')'
                l.append(v)
                start+=255
            result+='+'.join(l)
            pass
        result += ')'
        return result
    @staticmethod
    def _itemsFromStrings(strings,converter):
        return (converter(x).value() for x in strings)

if __name__ == "__main__":

    import unittest
    class TestList(_ValidatingParameterListBase):
        def _itemIsValid(self,item):
            return True
    class testMixins(unittest.TestCase):
        def testLargeList(self):
            #lists larger than 255 entries can not be initialized
            #using the constructor
            args = [i for i in xrange(0,300)]
            
            t = TestList(*args)
            pdump= t.dumpPython()
            class cms(object):
                def __init__(self):
                    self.TestList = TestList
            pythonized = eval( pdump, globals(),{'cms':cms()} )
            self.assertEqual(t,pythonized)
        def testUsingBlock(self):
            a = UsingBlock("a")
            self.assert_(isinstance(a, _ParameterTypeBase))
    unittest.main()
