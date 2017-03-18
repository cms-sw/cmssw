import inspect

class _ConfigureComponent(object):
    """Denotes a class that can be used by the Processes class"""
    def _isTaskComponent(self):
        return False

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
        self.__dict__["_isFrozen"] = False
        self.__isTracked = True
        self._isModified = False
    def isModified(self):
        return self._isModified
    def resetModified(self):
        self._isModified=False
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
    def isFrozen(self):
        return self._isFrozen 
    def setIsFrozen(self):
        self._isFrozen = True

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
        if value!=self._value:
            self._isModified=True
            self._value=value
    def configValue(self, options=PrintOptions()):
        return str(self._value)
    def pythonValue(self, options=PrintOptions()):
        return self.configValue(options)
    def __eq__(self,other):
        if isinstance(other,_SimpleParameterTypeBase):
            return self._value == other._value
        return self._value == other
    def __ne__(self,other):
        if isinstance(other,_SimpleParameterTypeBase):
            return self._value != other._value
        return self._value != other
    def __lt__(self,other):
        if isinstance(other,_SimpleParameterTypeBase):
            return self._value < other._value
        return self._value < other
    def __le__(self,other):
        if isinstance(other,_SimpleParameterTypeBase):
            return self._value <= other._value
        return self._value <= other
    def __gt__(self,other):
        if isinstance(other,_SimpleParameterTypeBase):
            return self._value > other._value
        return self._value > other
    def __ge__(self,other):
        if isinstance(other,_SimpleParameterTypeBase):
            return self._value >= other._value
        return self._value >= other


class UsingBlock(_SimpleParameterTypeBase):
    """For injection purposes, pretend this is a new parameter type
       then have a post process step which strips these out
    """
    def __init__(self,value, s='', loc=0, file=''):
        super(UsingBlock,self).__init__(value)
        self.s = s
        self.loc = loc
        self.file = file
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
    def dumpPython(self, options=PrintOptions()):
        if options.isCfg:
            return "process."+self.value()
        else:
            return self.value()


class _Parameterizable(object):
    """Base class for classes which allow addition of _ParameterTypeBase data"""
    def __init__(self,*arg,**kargs):
        self.__dict__['_Parameterizable__parameterNames'] = []
        self.__dict__["_isFrozen"] = False
        """The named arguments are the 'parameters' which are added as 'python attributes' to the object"""
        if len(arg) != 0:
            #raise ValueError("unnamed arguments are not allowed. Please use the syntax 'name = value' when assigning arguments.")
            for block in arg:
                if type(block).__name__ != "PSet":
                    raise ValueError("Only PSets can be passed as unnamed argument blocks.  This is a "+type(block).__name__)
                self.__setParameters(block.parameters_())
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

    def hasParameter(self, params):
        """
        _hasParameter_

        check that pset provided has the attribute chain
        specified.

        Eg, if params is [ 'attr1', 'attr2', 'attr3' ]
        check for pset.attr1.attr2.attr3

        returns True if parameter exists, False if not
        """
        return (self.getParameter(params) != None)

    def getParameter(self, params):
        """
        _getParameter_

        Retrieve the specified parameter from the PSet Provided
        given the attribute chain

        returns None if not found
        """
        lastParam = self
        # Don't accidentally iterate over letters in a string
        if type(params).__name__ == 'str':
            return getattr(self, params, None)
        for param in params:
            lastParam = getattr(lastParam, param, None)
            print str(lastParam)
            if lastParam == None:
                return None
        return lastParam

    def parameters_(self):
        """Returns a dictionary of copies of the user-set parameters"""
        import copy
        result = dict()
        for name in self.parameterNames_():
               result[name]=copy.deepcopy(self.__dict__[name])
        return result

    def __addParameter(self, name, value):
        if not isinstance(value,_ParameterTypeBase):
            self.__raiseBadSetAttr(name)
        if name in self.__dict__:
            message = "Duplicate insert of member " + name
            message += "\nThe original parameters are:\n"
            message += self.dumpPython() + '\n'
            raise ValueError(message)
        self.__dict__[name]=value
        self.__parameterNames.append(name)
        self._isModified = True

    def __setParameters(self,parameters):
        for name,value in parameters.iteritems():
            self.__addParameter(name, value)

    def __setattr__(self,name,value):
        #since labels are not supposed to have underscores at the beginning
        # I will assume that if we have such then we are setting an internal variable
        if self.isFrozen() and not (name in ["_Labelable__label","_isFrozen"] or name.startswith('_')): 
            message = "Object already added to a process. It is read only now\n"
            message +=  "    %s = %s" %(name, value)
            message += "\nThe original parameters are:\n"
            message += self.dumpPython() + '\n'           
            raise ValueError(message)
        # underscored names bypass checking for _ParameterTypeBase
        if name[0]=='_':
            super(_Parameterizable,self).__setattr__(name,value)
        elif not name in self.__dict__:
            self.__addParameter(name, value)
            self._isModified = True
        else:
            # handle the case where users just replace with a value, a = 12, rather than a = cms.int32(12)
            if isinstance(value,_ParameterTypeBase):
                self.__dict__[name] = value
            else:
                self.__dict__[name].setValue(value)
            self._isModified = True

    def isFrozen(self):
        return self._isFrozen
    def setIsFrozen(self):
        self._isFrozen = True
        for name in self.parameterNames_():
            self.__dict__[name].setIsFrozen() 
    def __delattr__(self,name):
        if self.isFrozen():
            raise ValueError("Object already added to a process. It is read only now")
        super(_Parameterizable,self).__delattr__(name)
        self.__parameterNames.remove(name)
    @staticmethod
    def __raiseBadSetAttr(name):
        raise TypeError(name+" does not already exist, so it can only be set to a CMS python configuration type")
    def dumpPython(self, options=PrintOptions()):
        sortedNames = sorted(self.parameterNames_())
        if len(sortedNames) > 200:
        #Too many parameters for a python function call
        # The solution is to create a temporary dictionary which
        # is constructed by concatenating long lists (with maximum
        # 200 entries each) together.
        # This looks like
        #  **dict( [(...,...), ...] + [...] + ... )
            others = []
            usings = []
            for name in sortedNames:
                param = self.__dict__[name]
                # we don't want minuses in names
                name2 = name.replace('-','_')
                options.indent()
                #_UsingNodes don't get assigned variables
                if name.startswith("using_"):
                    usings.append(options.indentation()+param.dumpPython(options))
                else:
                    others.append((name2, param.dumpPython(options)))
                options.unindent()

            resultList = ',\n'.join(usings)
            longOthers = options.indentation()+"**dict(\n"
            options.indent()
            longOthers += options.indentation()+"[\n"
            entriesInList = 0
            options.indent()
            for n,v in others:
                entriesInList +=1
                if entriesInList > 200:
                    #need to start a new list
                    options.unindent()
                    longOthers += options.indentation()+"] +\n"+options.indentation()+"[\n"
                    entriesInList = 0
                    options.indent()
                longOthers += options.indentation()+'("'+n+'" , '+v+' ),\n'
            
            longOthers += options.indentation()+"]\n"
            options.unindent()
            longOthers +=options.indentation()+")\n"
            options.unindent()
            ret = []
            if resultList:
                ret.append(resultList)
            if longOthers:
                ret.append(longOthers)
            return ",\n".join(ret)
        #Standard case, small number of parameters
        others = []
        usings = []
        for name in sortedNames:
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
        saveOrigin(self, 1) 
    def _place(self,name,proc):
        self._placeImpl(name,proc)
    def type_(self):
        """returns the type of the object, e.g. 'FooProducer'"""
        return self.__type
    def copy(self):
        returnValue =_TypedParameterizable.__new__(type(self))
        params = self.parameters_()
        args = list()
        if len(params) == 0:
            args.append(None)
        returnValue.__init__(self.__type,*args,
                             **params)
        returnValue._isModified = self._isModified
        return returnValue
    def clone(self, *args, **params):
        """Copies the object and allows one to modify the parameters of the clone.
        New parameters may be added by specify the exact type
        Modifying existing parameters can be done by just specifying the new
          value without having to specify the type.
        A parameter may be removed from the clone using the value None.
           #remove the parameter foo.fred
           mod.toModify(foo, fred = None)
        A parameter embedded within a PSet may be changed via a dictionary
           #change foo.fred.pebbles to 3 and foo.fred.friend to "barney"
           mod.toModify(foo, fred = dict(pebbles = 3, friend = "barney)) )
        """
        returnValue =_TypedParameterizable.__new__(type(self))
        myparams = self.parameters_()
        if len(myparams) == 0 and len(params) and len(args):
            args.append(None)
        
        _modifyParametersFromDict(myparams, params, self._Parameterizable__raiseBadSetAttr)

        returnValue.__init__(self.__type,*args,
                             **myparams)
        returnValue._isModified = False
        returnValue._isFrozen = False
        saveOrigin(returnValue, 1)
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
        result = "cms."+str(type(self).__name__)+'("'+self.type_()+'"'
        nparam = len(self.parameterNames_())
        if nparam == 0:
            result += ")\n"
        else:
            result += ",\n"+_Parameterizable.dumpPython(self,options)+options.indentation() + ")\n"
        return result

    def dumpPythonAttributes(self, myname, options):
        """ dumps the object with all attributes declared after the constructor"""
        result = ""
        for name in sorted(self.parameterNames_()):
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
        newpset.addString(True, "@module_edm_type", type(self).__name__)
        self.insertContentsInto(newpset)
        parameterSet.addPSet(True, self.nameInProcessDesc_(myname), newpset)



class _Labelable(object):
    """A 'mixin' used to denote that the class can be paired with a label (e.g. an EDProducer)"""
    def label_(self):
        if not hasattr(self, "_Labelable__label"):
           raise RuntimeError("module has no label.  Perhaps it wasn't inserted into the process?")
        return self.__label
    def hasLabel_(self):
        return hasattr(self, "_Labelable__label") and self.__label is not None
    def setLabel(self,label):
        if self.hasLabel_() :
            if self.label_() != label and label is not None :
                msg100 = "Attempting to change the label of a Labelable object, possibly an attribute of the Process\n"
                msg101 = "Old label = "+self.label_()+"  New label = "+label+"\n"
                msg102 = "Type = "+str(type(self))+"\n"
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
        self.__label = label
    def label(self):
        #print "WARNING: _Labelable::label() needs to be changed to label_()"
        return self.__label
    def __str__(self):
        #this is probably a bad idea
        # I added this so that when we ask a path to print
        # we will see the label that has been assigned
        return str(self.__label)
    def dumpSequenceConfig(self):
        return str(self.__label)
    def dumpSequencePython(self, options=PrintOptions()):
        return 'process.'+str(self.__label)
    def _findDependencies(self,knownDeps,presentDeps):
        #print 'in labelled'
        myDeps=knownDeps.get(self.label_(),None)
        if myDeps!=None:
            if presentDeps != myDeps:
                raise RuntimeError("the module "+self.label_()+" has two dependencies \n"
                                   +str(presentDeps)+"\n"
                                   +str(myDeps)+"\n"
                                   +"Please modify sequences to rectify this inconsistency")
        else:
            myDeps=set(presentDeps)
            knownDeps[self.label_()]=myDeps
        presentDeps.add(self.label_())


class _Unlabelable(object):
    """A 'mixin' used to denote that the class can be used without a label (e.g. a Service)"""
    pass

class _ValidatingListBase(list):
    """Base class for a list which enforces that its entries pass a 'validity' test"""
    def __init__(self,*arg,**args):        
        super(_ValidatingListBase,self).__init__(arg)
        if 0 != len(args):
            raise SyntaxError("named arguments ("+','.join([x for x in args])+") passsed to "+str(type(self)))
        if not self._isValid(iter(self)):
            raise TypeError("wrong types ("+','.join([str(type(value)) for value in iter(self)])+
                            ") added to "+str(type(self)))
    def __setitem__(self,key,value):
        if isinstance(key,slice):
            if not self._isValid(value):
                raise TypeError("wrong type being inserted into this container "+self._labelIfAny())
        else:
            if not self._itemIsValid(value):
                raise TypeError("can not insert the type "+str(type(value))+" in container "+self._labelIfAny())
        super(_ValidatingListBase,self).__setitem__(key,value)
    def _isValid(self,seq):
        # see if strings get reinterpreted as lists
        if isinstance(seq, str):
            return False
        for item in seq:
            if not self._itemIsValid(item):
                return False
        return True
    def append(self,x):
        if not self._itemIsValid(x):
            raise TypeError("wrong type being appended to container "+self._labelIfAny())
        super(_ValidatingListBase,self).append(x)
    def extend(self,x):
        if not self._isValid(x):
            raise TypeError("wrong type being extended to container "+self._labelIfAny())
        super(_ValidatingListBase,self).extend(x)
    def __add__(self,rhs):
        if not self._isValid(rhs):
            raise TypeError("wrong type being added to container "+self._labelIfAny())
        import copy
        value = copy.copy(self)
        value.extend(rhs)
        return value
    def insert(self,i,x):
        if not self._itemIsValid(x):
            raise TypeError("wrong type being inserted to container "+self._labelIfAny())
        super(_ValidatingListBase,self).insert(i,x)
    def _labelIfAny(self):
        result = type(self).__name__
        if hasattr(self, '__label'):
            result += ' ' + self.__label
        return result

class _ValidatingParameterListBase(_ValidatingListBase,_ParameterTypeBase):
    def __init__(self,*arg,**args):        
        _ParameterTypeBase.__init__(self)
        if len (arg) == 1 and not isinstance(arg[0],str):
            try:
                arg = iter(arg[0])
            except TypeError:
                pass
        super(_ValidatingParameterListBase,self).__init__(*arg,**args)
    def value(self):
        return list(self)
    def setValue(self,v):
        self[:] = []
        self.extend(v)
        self._isModified=True
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
        n = len(self)
        if n>=256:
            #wrap in a tuple since they don't have a size constraint
            result+=" ("
        indented = False
        for i, v in enumerate(self):
            if i == 0:
                if hasattr(self, "_nPerLine"):
                    nPerLine = self._nPerLine
                else:
                    nPerLine = 5
            else:
                if not indented:
                    indented = True
                    options.indent()
                result += ', '
                if i % nPerLine == 0:
                    result += '\n'+options.indentation()
            result += self.pythonValueForItem(v,options)
        if indented:
            options.unindent()
        #result+=', '.join((self.pythonValueForItem(v,options) for v in iter(self)))
        if n>=256:
            result +=' ) '
        result += ')'
        return result            
    @staticmethod
    def _itemsFromStrings(strings,converter):
        return (converter(x).value() for x in strings)

def saveOrigin(obj, level):
    #frame = inspect.stack()[level+1]
    frame = inspect.getframeinfo(inspect.currentframe(level+1))
    # not safe under old python versions
    #obj._filename = frame.filename
    #obj._lineNumber = frame.lineno
    obj._filename = frame[0]
    obj._lineNumber = frame[1]

def _modifyParametersFromDict(params, newParams, errorRaiser, keyDepth=""):
    if len(newParams):
        #need to treat items both in params and myparams specially
        for key,value in newParams.iteritems():
            if key in params:
                if value is None:
                    del params[key]
                elif isinstance(value, dict):
                    if isinstance(params[key],_Parameterizable):
                        pset = params[key]
                        p =pset.parameters_()
                        _modifyParametersFromDict(p,
                                                  value,errorRaiser,
                                                  ("%s.%s" if type(key)==str else "%s[%s]")%(keyDepth,key))
                        for k,v in p.iteritems():
                            setattr(pset,k,v)
                    elif isinstance(params[key],_ValidatingParameterListBase):
                        if any(type(k) != int for k in value.keys()):
                            raise TypeError("Attempted to change a list using a dict whose keys are not integers")
                        plist = params[key]
                        if any((k < 0 or k >= len(plist)) for k in value.keys()):
                            raise IndexError("Attempted to set an index which is not in the list")
                        p = dict(enumerate(plist))
                        _modifyParametersFromDict(p,
                                                  value,errorRaiser,
                                                  ("%s.%s" if type(key)==str else "%s[%s]")%(keyDepth,key))
                        for k,v in p.iteritems():
                            plist[k] = v
                    else:
                        raise ValueError("Attempted to change non PSet value "+keyDepth+" using a dictionary")
                elif isinstance(value,_ParameterTypeBase) or (type(key) == int):
                    params[key] = value
                else:
                    params[key].setValue(value)
            else:
                if isinstance(value,_ParameterTypeBase):
                    params[key]=value
                else:
                    errorRaiser(key)


if __name__ == "__main__":

    import unittest
    class TestList(_ValidatingParameterListBase):
        def _itemIsValid(self,item):
            return True
    class testMixins(unittest.TestCase):
        def testListConstruction(self):
            t = TestList(1)
            self.assertEqual(t,[1])
            t = TestList((1,))
            self.assertEqual(t,[1])
            t = TestList("one")
            self.assertEqual(t,["one"])
            t = TestList( [1,])
            self.assertEqual(t,[1])
            t = TestList( (x for x in [1]) )
            self.assertEqual(t,[1])

            t = TestList(1,2)
            self.assertEqual(t,[1,2])
            t = TestList((1,2))
            self.assertEqual(t,[1,2])
            t = TestList("one","two")
            self.assertEqual(t,["one","two"])
            t = TestList(("one","two"))
            self.assertEqual(t,["one","two"])
            t = TestList( [1,2])
            self.assertEqual(t,[1,2])
            t = TestList( (x for x in [1,2]) )
            self.assertEqual(t,[1,2])
            t = TestList( iter((1,2)) )
            self.assertEqual(t,[1,2])
            
            
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
        def testCopy(self):
            class __Test(_TypedParameterizable):
                pass
            class __TestType(_SimpleParameterTypeBase):
                def _isValid(self,value):
                    return True
            a = __Test("MyType",t=__TestType(1), u=__TestType(2))
            b = a.copy()
            self.assertEqual(b.t.value(),1)
            self.assertEqual(b.u.value(),2)
        def testClone(self):
            class __Test(_TypedParameterizable):
                pass
            class __TestType(_SimpleParameterTypeBase):
                def _isValid(self,value):
                    return True
            class __PSet(_ParameterTypeBase,_Parameterizable):
                def __init__(self,*arg,**args):
                    #need to call the inits separately
                    _ParameterTypeBase.__init__(self)
                    _Parameterizable.__init__(self,*arg,**args)
            a = __Test("MyType",
                       t=__TestType(1),
                       u=__TestType(2),
                       w = __TestType(3),
                       x = __PSet(a = __TestType(4),
                                  b = __TestType(6),
                                  c = __PSet(gamma = __TestType(5))))
            b = a.clone(t=3,
                        v=__TestType(4),
                        w= None,
                        x = dict(a = 7,
                                 c = dict(gamma = 8),
                                 d = __TestType(9)))
            self.assertEqual(a.t.value(),1)
            self.assertEqual(a.u.value(),2)
            self.assertEqual(b.t.value(),3)
            self.assertEqual(b.u.value(),2)
            self.assertEqual(b.v.value(),4)
            self.assertEqual(b.x.a.value(),7)
            self.assertEqual(b.x.b.value(),6)
            self.assertEqual(b.x.c.gamma.value(),8)
            self.assertEqual(b.x.d.value(),9)
            self.assertEqual(hasattr(b,"w"), False)
            self.assertRaises(TypeError,a.clone,None,**{"v":1})
        def testModified(self):
            class __TestType(_SimpleParameterTypeBase):
                def _isValid(self,value):
                    return True
            a = __TestType(1)
            self.assertEqual(a.isModified(),False)
            a.setValue(1)
            self.assertEqual(a.isModified(),False)
            a.setValue(2)
            self.assertEqual(a.isModified(),True)
            a.resetModified()
            self.assertEqual(a.isModified(),False)
        def testLargeParameterizable(self):
            class tLPTest(_TypedParameterizable):
                pass
            class tLPTestType(_SimpleParameterTypeBase):
                def _isValid(self,value):
                    return True
            class __DummyModule(object):
                def __init__(self):
                    self.tLPTest = tLPTest
                    self.tLPTestType = tLPTestType
            p = tLPTest("MyType",** dict( [ ("a"+str(x), tLPTestType(x)) for x in xrange(0,300) ] ) )
            #check they are the same
            self.assertEqual(p.dumpPython(), eval(p.dumpPython(),{"cms": __DummyModule()}).dumpPython())
    unittest.main()
