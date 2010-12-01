
from Mixins import _ConfigureComponent, PrintOptions
from Mixins import _Labelable, _Unlabelable
from Mixins import _ValidatingParameterListBase
from ExceptionHandling import *

class _HardDependency(object):
    """Information relevant for when a hard dependency, 
       which uses the * operator, is found"""
    def __init__(self, sequenceName, depSet):
        self.sequenceName = sequenceName
        self.depSet = depSet

class _Sequenceable(object):
    """Denotes an object which can be placed in a sequence"""
    def __init__(self):
        pass
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
            raise KeyError("no "+str(type(self))+" with id "+str(id(self))+" found")
    def _replace(self, original, replacement):
        pass
    def _remove(self, original):
        """Remove 'original'. Return can be
             (_Sequenceable, True ): module was found and removed, this is the new non-empty sequence.
             (_Sequenceable, False): module was not found, this is the original sequence (that is, 'self')
             (None,          True ): the module was found and removed, the result is an empty sequence."""
        return (self, False)
    def resolve(self, processDict):
        return self
    def isOperation(self):
        """Returns True if the object is an operator (e.g. *,+ or !) type"""
        return False
    def _visitSubNodes(self,visitor):
        pass
    def visitNode(self,visitor):
        visitor.enter(self)
        self._visitSubNodes(visitor)
        visitor.leave(self)
    def findHardDependencies(self, sequenceName, dependencyDict):
        pass

def _checkIfSequenceable(caller, v):
    if not isinstance(v,_Sequenceable):
        typename = format_typename(caller)
        msg = format_outerframe(2)
        msg += "%s only takes arguments of types which are allowed in a sequence, but was given:\n" %typename
        msg +=format_typename(v)
        msg +="\nPlease remove the problematic object from the argument list"
        raise TypeError(msg)

class _SequenceLeaf(_Sequenceable):
    def __init__(self):
        pass

class _ModuleSequenceType(_ConfigureComponent, _Labelable):
    """Base class for classes which define a sequence of modules"""
    def __init__(self,*arg, **argv):
        self.__dict__["_isFrozen"] = False
        self._seq = None
        if len(arg) > 1:
            typename = format_typename(self)
            msg = format_outerframe(2) 
            msg += "%s takes exactly one input value. But the following ones are given:\n" %typename
            for item,i in zip(arg, xrange(1,20)):
                msg += "    %i) %s \n"  %(i, item._errorstr())
            msg += "Maybe you forgot to combine them via '*' or '+'."     
            raise TypeError(msg)
        if len(arg)==1:
            _checkIfSequenceable(self, arg[0])
            self._seq = arg[0]
        self._isModified = False
    def isFrozen(self):
        return self._isFrozen
    def setIsFrozen(self):
        self._isFrozen = True 
    def _place(self,name,proc):
        self._placeImpl(name,proc)
    def __imul__(self,rhs):
        _checkIfSequenceable(self, rhs)
        if self._seq is None:
            self._seq = rhs
        else:
            self._seq = _SequenceOpAids(self._seq,rhs)
        return self
    def __iadd__(self,rhs):
        _checkIfSequenceable(self, rhs)
        if self._seq is None:
            self._seq = rhs
        else:
            self._seq = _SequenceOpFollows(self._seq,rhs)
        return self
    def __str__(self):
        return str(self._seq)
    def dumpConfig(self, options):
        return '{'+self._seq.dumpSequenceConfig()+'}\n'
    def dumpPython(self, options):
        s=''
        if self._seq is not None:
            s =self._seq.dumpSequencePython()
        return 'cms.'+type(self).__name__+'('+s+')\n'
    def dumpSequencePython(self):
        # only dump the label, if possible
        if self.hasLabel_():
            return _Labelable.dumpSequencePython(self)
        else:
            # dump it verbose
            if self._seq is None:
                return ''
            return '('+self._seq.dumpSequencePython()+')'
    def __repr__(self):
        s = ''
        if self._seq is not None:
           s = str(self._seq)
        return "cms."+type(self).__name__+'('+s+')\n'
    def copy(self):
        returnValue =_ModuleSequenceType.__new__(type(self))
        if self._seq is not None:
            returnValue.__init__(self._seq)
        else:
            returnValue.__init__()
        return returnValue
    def copyAndExclude(self,listOfModulesToExclude):
        """Returns a copy of the sequence which exlcudes those module in 'listOfModulesToExclude'"""
        v = _CopyAndExcludeSequenceVisitor(listOfModulesToExclude)
        self.visit(v)
        result = self.__new__(type(self))
        result.__init__(v.result())
        return result
    def expandAndClone(self):
        visitor = ExpandVisitor(type(self))
        self.visit(visitor)
        return visitor.result()
    def _postProcessFixup(self,lookuptable):
        self._seq = self._seq._clonesequence(lookuptable)
        return self
    def replace(self, original, replacement):
        if not isinstance(original,_Sequenceable) or not isinstance(replacement,_Sequenceable):
           raise TypeError("replace only works with sequenceable objects")
        else:
           self._replace(original, replacement)
    def _replace(self, original, replacement):
        if self._seq == original:
            self._seq = replacement
        else:
            if self._seq is not None:
                self._seq._replace(original,replacement)
    def remove(self, something):
        """Remove the leftmost occurrence of 'something' (a sequence or a module)
           It will give an error if removing 'something' leaves a cms.Sequence empty.
           Returns 'True' if the module has been removed, False if it was not found"""
        (seq, found) = self._remove(something)
        return found
    def _remove(self, original):
        if (self._seq == original):
            self._seq = None
            return (None, True)
        (self._seq, found) = self._seq._remove(original);
        return (self, found)
    def resolve(self, processDict):
        if self._seq is not None:
            self._seq = self._seq.resolve(processDict)
        return self
    def __setattr__(self,name,value):
        if not name.startswith("_"):
            raise AttributeError, "You cannot set parameters for sequence like objects."
        else:
            self.__dict__[name] = value
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
        if self._seq is not None:
            self._seq._findDependencies(knownDeps,presentDeps)
    def moduleDependencies(self):
        deps = dict()
        self._findDependencies(deps,set())
        return deps
    def nameInProcessDesc_(self, myname):
        return myname
    def insertInto(self, parameterSet, myname, processDict):
        # represented just as a list of names in the ParameterSet
        l = []
        #resolver = ResolveVisitor(processDict)
        lister = DecoratedNodeNameVisitor(l)
        #self.visit(resolver)
        self.resolve(processDict)
        self.visit(lister)
        parameterSet.addVString(True, myname, l)
    def visit(self,visitor):
        """Passes to visitor's 'enter' and 'leave' method each item describing the module sequence.
        If the item contains 'sub' items then visitor will see those 'sub' items between the
        item's 'enter' and 'leave' calls.
        """
        if self._seq is not None:
            self._seq.visitNode(visitor)
    def findHardDependencies(self, sequenceName, dependencyDict):
        if self._seq is not None and self.hasLabel_():
            self._seq.findHardDependencies(self.label_(), dependencyDict)


class _SequenceOperator(_Sequenceable):
    """Used in the expression tree for a sequence"""
    def __init__(self, left, right):
        _checkIfSequenceable(self,left)
        _checkIfSequenceable(self,right)
        self._left = left
        self._right = right
    def __str__(self):
        returnValue = self._dumpChild(self._left, str(self._left))
        returnValue +=self._pySymbol
        returnValue +=self._dumpChild(self._right, str(self._right))
        return returnValue
    def dumpSequenceConfig(self):
        returnValue = self._dumpChild(self._left, self._left.dumpSequenceConfig())
        returnValue +=self._cfgSymbol
        returnValue +=self._dumpChild(self._right, self._right.dumpSequenceConfig())
        return returnValue
    def dumpSequencePython(self):
        returnValue = self._dumpChild(self._left, self._left.dumpSequencePython())
        returnValue +=self._pySymbol
        returnValue +=self._dumpChild(self._right, self._right.dumpSequencePython())
        return returnValue
    def _dumpChild(self, child, dump):
        returnValue = dump
        # see if it needs parentheses for precedence
        if isinstance(child, _SequenceOperator) and (child._precedence() < self._precedence()):
           returnValue = '('+returnValue+')'
        return returnValue
    def _clonesequence(self, lookuptable):
        return type(self)(self._left._clonesequence(lookuptable),self._right._clonesequence(lookuptable))
    def _replace(self, original, replacement):
        if self._left == original:
            self._left = replacement
        else:
            self._left._replace(original, replacement)
        if self._right == original:
            self._right = replacement
        else:
            self._right._replace(original, replacement)                    
    def _remove(self, original):
        if self._left == original:  return (self._right, True) # left IS what we want to remove
        (self._left, found) = self._left._remove(original)     # otherwise clean left
        if (self._left == None):    return (self._right, True) # left is empty after cleaning
        if found:                   return (self, True)        # found on left, don't clean right
        if self._right == original: return (self._left, True)  # right IS what we want to remove
        (self._right, found) = self._right._remove(original)   # otherwise clean right
        if (self._right == None):   return (self._left, True)  # right is empty after cleaning
        return (self,found)                                    # return what we found
    def resolve(self, processDict):
        self._left = self._left.resolve(processDict)
        self._right = self._right.resolve(processDict)
        return self
    def isOperation(self):
        return True
    def _visitSubNodes(self,visitor):
        self._left.visitNode(visitor)
        self._right.visitNode(visitor)
    def _precedence(self):
        """Precedence order for this operation, the larger the value the higher the precedence"""
        raise RuntimeError("_precedence must be overwritten by inheriting classes")
        return 0


class _SequenceOpAids(_SequenceOperator):
    """Used in the expression tree for a sequence as a stand in for the ',' operator"""
    def __init__(self, left, right):
        _SequenceOperator.__init__(self, left, right)
        self._cfgSymbol = ','
        self._pySymbol = '*'
    def _findDependencies(self,knownDeps,presentDeps):
        #do left first and then right since right depends on left
        self._left._findDependencies(knownDeps,presentDeps)
        self._right._findDependencies(knownDeps,presentDeps)
    def _precedence(self):
        return 2
    def findHardDependencies(self, sequenceName, dependencyDict):
        # everything on the RHS depends on everything on the LHS
        rhs = set()
        moduleNames = NodeNameVisitor(rhs)
        self._right.visitNode(moduleNames)
        lhs = set()
        moduleNames = NodeNameVisitor(lhs)
        self._left.visitNode(moduleNames)
        dep = _HardDependency(sequenceName, lhs)
        for rhsmodule in rhs:
            if not rhsmodule in dependencyDict:
                dependencyDict[rhsmodule] = list()
            dependencyDict[rhsmodule].append(dep)
        self._left.findHardDependencies(sequenceName, dependencyDict)
        self._right.findHardDependencies(sequenceName, dependencyDict)

class _SequenceOpFollows(_SequenceOperator):
    """Used in the expression tree for a sequence as a stand in for the '&' operator"""
    def __init__(self, left, right):
        _SequenceOperator.__init__(self, left, right)
        self._cfgSymbol = '&'
        self._pySymbol = '+'
    def _findDependencies(self,knownDeps,presentDeps):
        oldDepsL = presentDeps.copy()
        oldDepsR = presentDeps.copy()
        self._left._findDependencies(knownDeps,oldDepsL)
        self._right._findDependencies(knownDeps,oldDepsR)
        end = len(presentDeps)
        presentDeps.update(oldDepsL)
        presentDeps.update(oldDepsR)
    def _precedence(self):
        return 1
    def findHardDependencies(self, sequenceName, dependencyDict):
        self._left.findHardDependencies(sequenceName, dependencyDict)
        self._right.findHardDependencies(sequenceName, dependencyDict)


class _UnarySequenceOperator(_Sequenceable):
    """For ~ and - operators"""
    def __init__(self, operand):
       self._operand = operand
       if isinstance(operand, _ModuleSequenceType):
           raise RuntimeError("This operator cannot accept a sequence")
    def __eq__(self, other):
        # allows replace(~a, b)
        return type(self) == type(other) and self._operand==other._operand
    def __ne__(self, other):
        return not self.__eq__(other)
    def _findDependencies(self,knownDeps, presentDeps):
        self._operand._findDependencies(knownDeps, presentDeps)
    def _clonesequence(self, lookuptable):
        return type(self)(self._operand._clonesequence(lookuptable))
    def _replace(self, original, replacement):
        if self._operand == original:
            self._operand = replacement
        else:
            self._operand._replace(original, replacement)
    def _remove(self, original):
        if (self._operand == original): return (None, True)
        (self._operand, found) = self._operand._remove(original)
        if self._operand == None: return (None, True)
        return (self, found)
    def resolve(self, processDict):
        self._operand = self._operand.resolve(processDict)
        return self
    def isOperation(self):
        return True
    def _visitSubNodes(self,visitor):
        self._operand.visitNode(visitor)
    def findHardDependencies(self, sequenceName, dependencyDict):
        pass


class _SequenceNegation(_UnarySequenceOperator):
    """Used in the expression tree for a sequence as a stand in for the '!' operator"""
    def __init__(self, operand):
        super(_SequenceNegation,self).__init__(operand)
    def __str__(self):
        return '~%s' %self._operand
    def dumpSequenceConfig(self):
        return '!%s' %self._operand.dumpSequenceConfig()
    def dumpSequencePython(self):
        return '~%s' %self._operand.dumpSequencePython()

class _SequenceIgnore(_UnarySequenceOperator):
    """Used in the expression tree for a sequence as a stand in for the '-' operator"""
    def __init__(self, operand):
        super(_SequenceIgnore,self).__init__(operand)
    def __str__(self):
        return 'cms.ignore(%s)' %self._operand
    def dumpSequenceConfig(self):
        return '-%s' %self._operand.dumpSequenceConfig()
    def dumpSequencePython(self):
        return 'cms.ignore(%s)' %self._operand.dumpSequencePython()

def ignore(seq):
    """The EDFilter passed as an argument will be run but its filter value will be ignored
    """
    return _SequenceIgnore(seq)

class Path(_ModuleSequenceType):
    def __init__(self,*arg,**argv):
        super(Path,self).__init__(*arg,**argv)
    def _placeImpl(self,name,proc):
        proc._placePath(name,self)

class EndPath(_ModuleSequenceType):
    def __init__(self,*arg,**argv):
        super(EndPath,self).__init__(*arg,**argv)
    def _placeImpl(self,name,proc):
        proc._placeEndPath(name,self)

class Sequence(_ModuleSequenceType,_Sequenceable):
    def __init__(self,*arg,**argv):
        super(Sequence,self).__init__(*arg,**argv)
    def _placeImpl(self,name,proc):
        proc._placeSequence(name,self)
    def _clonesequence(self, lookuptable):
        if id(self) not in lookuptable:
            #for sequences held by sequences we need to clone
            # on the first reference
            if self._seq is not None:
                clone = type(self)(self._seq._clonesequence(lookuptable))
            else:
                clone = type(self)()
            lookuptable[id(self)]=clone
            lookuptable[id(clone)]=clone
        return lookuptable[id(self)]
    def _visitSubNodes(self,visitor):
        self.visit(visitor)

class SequencePlaceholder(_Sequenceable):
    def __init__(self, name):
        self._name = name
    def _placeImpl(self,name,proc):
        pass
    def __str__(self):
        return self._name
    def insertInto(self, parameterSet, myname):
        raise RuntimeError("The SequencePlaceholder "+self._name
                           +" was never overridden")
    def resolve(self, processDict):
        if not self._name in processDict:
            print str(processDict.keys())
            raise RuntimeError("The SequencePlaceholder "+self._name+ " cannot be resolved.i\n Known keys are:"+str(processDict.keys()))
        return  processDict[self._name]

    def _clonesequence(self, lookuptable):
        if id(self) not in lookuptable:
            #for sequences held by sequences we need to clone
            # on the first reference
            clone = type(self)(self._name)
            lookuptable[id(self)]=clone
            lookuptable[id(clone)]=clone
        return lookuptable[id(self)]
    def copy(self):
        returnValue =SequencePlaceholder.__new__(type(self))
        returnValue.__init__(self._name)
        return returnValue
    def dumpSequenceConfig(self):
        return 'cms.SequencePlaceholder("%s")' %self._name
    def dumpSequencePython(self):
        return 'cms.SequencePlaceholder("%s")'%self._name
    def dumpPython(self, options):
        result = 'cms.SequencePlaceholder(\"'
        if options.isCfg:
           result += 'process.'
        result += +self._name+'\")\n'
    

class Schedule(_ValidatingParameterListBase,_ConfigureComponent,_Unlabelable):
    def __init__(self,*arg,**argv):
        super(Schedule,self).__init__(*arg,**argv)
    @staticmethod
    def _itemIsValid(item):
        return isinstance(item,Path) or isinstance(item,EndPath)
    def copy(self):
        import copy
        return copy.copy(self)
    def _place(self,label,process):
        process.setPartialSchedule_(self,label)
    def moduleNames(self):
        result = set()
        visitor = NodeNameVisitor(result)
        for seq in self:
            seq.visit(visitor)
        return result
    def enforceDependencies(self):
        # I don't think we need the processDict
        processDict = dict()
        dependencyDict = dict()
        names = set()
        namesVisitor = NodeNameVisitor(names)
        ok = True
        errors = list()
        for seq in self:
            seq.visit(namesVisitor)
            seq.findHardDependencies('schedule', dependencyDict)
        # dependencyDict is (label, list of _HardDependency objects,
        # where a _HardDependency contains a set of strings from one sequence
        for label, depList in dependencyDict.iteritems():
            # see if it's in 
            try:
                thisPos = names.index(label)
                # we had better find all the dependencies
                for hardDep in depList:
                    for dep in hardDep.depsSet:
                        if names[0:thisPos].count(dep) == 0:
                            ok = False 
                            message = "WARNING:"+label+" depends on "+dep+", as declared in " \
                                      + hardDep.sequenceName+", but not found in schedule"
                            print message
            except:  
                # can't find it?  No big deal.
                pass


class SequenceVisitor(object):
    def __init__(self,d):
        self.deps = d
    def enter(self,visitee):
        if isinstance(visitee,Sequence):
            self.deps.append(visitee)
        pass
    def leave(self,visitee):
        pass

class ModuleNodeVisitor(object):
    def __init__(self,l):
        self.l = l
    def enter(self,visitee):
        if isinstance(visitee,_SequenceLeaf):
            self.l.append(visitee)
        pass
    def leave(self,visitee):
        pass


class NodeNameVisitor(object):
    """ takes a set as input"""
    def __init__(self,l):
        self.l = l
    def enter(self,visitee):
        if isinstance(visitee,_SequenceLeaf):
            self.l.add(visitee.label_())
        pass
    def leave(self,visitee):
        pass


class ExpandVisitor(object):
    """ Expands the sequence into leafs and UnaryOperators """
    def __init__(self, type):
        self._type = type
        self.l = []
    def enter(self,visitee):
        if isinstance(visitee,_SequenceLeaf):
            self.l.append(visitee)
    def leave(self, visitee):
        if isinstance(visitee,_UnarySequenceOperator):
            self.l[-1] = visitee
    def result(self):
        # why doesn't (sum(self.l) work?
        seq = self.l[0]
        if len(self.l) > 1:
            for el in self.l[1:]:
                seq += el
        return self._type(seq)

    

class DecoratedNodeNameVisitor(object):
    """ Adds any '!' or '-' needed.  Takes a list """
    def __init__(self,l):
        self.l = l
    def enter(self,visitee):
        if isinstance(visitee,_SequenceLeaf):
            if hasattr(visitee, "_Labelable__label"):
                self.l.append(visitee.label_())
            else:
                error = "An object in a sequence was not found in the process\n"
                if hasattr(visitee, "_filename"):
                    error += "From file " + visitee._filename
                else:
                    error += "Dump follows\n" + repr(visitee)
                raise RuntimeError(error)
    def leave(self,visitee):
        if isinstance(visitee,_UnarySequenceOperator):
           self.l[-1] = visitee.dumpSequenceConfig()


class ResolveVisitor(object):
    """ Doesn't seem to work """
    def __init__(self,processDict):
        self.processDict = processDict
    def enter(self,visitee):
        if isinstance(visitee, SequencePlaceholder):
            if not visitee._name in self.processDict:
                print str(self.processDict.keys())
                raise RuntimeError("The SequencePlaceholder "+visitee._name+ " cannot be resolved.\n Known keys are:"+str(self.processDict.keys()))
            visitee = self.processDict[visitee._name]
    def leave(self,visitee):
       if isinstance(visitee, SequencePlaceholder):
           pass

class _CopyAndExcludeSequenceVisitor(object):
   """Traverses a Sequence and constructs a new sequence which does not contain modules from the specified list"""
   def __init__(self,modulesToRemove):
       self.__modulesToIgnore = modulesToRemove
       self.__stack = list()
       self.__result = None
   def enter(self,visitee):
       if len(self.__stack) > 0:
           #add visitee to its parent's stack entry
           self.__stack[-1].append([visitee,False])
       if isinstance(visitee,_SequenceLeaf):
           if visitee in self.__modulesToIgnore:
               self.__stack[-1][-1]=[None,True]
       else:
           #need to add a stack entry to keep track of children
           self.__stack.append(list())
   def leave(self,visitee):
       node = visitee
       if not isinstance(visitee,_SequenceLeaf):
           #were any children changed?
           l = self.__stack[-1]
           changed = False
           countNulls = 0
           nonNulls = list()
           for c in l:
               if c[1] == True:
                   changed = True
               if c[0] is None:
                   countNulls +=1
               else:
                   nonNulls.append(c[0])
           if changed:
               if countNulls != 0:
                   #this node must go away
                   if len(nonNulls) == 0:
                       #all subnodes went away 
                       node = None
                   else:
                       #we assume only possible to have one non null
                       # so replace this node with that one child
                       node = nonNulls[0]
               else:
                   #some child was changed so we need to clone
                   # this node and replace it with one that holds 
                   # the new child(ren)
                   children = [x[0] for x in l ]
                   if not isinstance(visitee,Sequence):
                       node = visitee.__new__(type(visitee))
                       node.__init__(*children)
                   else:
                       node = nonNulls[0]
       if node != visitee:
           #we had to replace this node so now we need to 
           # change parent's stack entry as well
           if len(self.__stack) > 1:
               p = self.__stack[-2]
               #find visitee and replace
               for i,c in enumerate(p):
                   if c[0]==visitee:
                       c[0]=node
                       c[1]=True
                       break
       self.__result = node
       if not isinstance(visitee,_SequenceLeaf):
           self.__stack = self.__stack[:-1]        
   def result(self):
       return self.__result



if __name__=="__main__":
    import unittest
    class DummyModule(_Labelable, _SequenceLeaf):
        def __init__(self,name):
            self.setLabel(name)
    class TestModuleCommand(unittest.TestCase):
        def setUp(self):
            """Nothing to do """
            pass
        def testDumpPython(self):
            a = DummyModule("a")
            b = DummyModule('b')
            p = Path((a*b))
            #print p.dumpConfig('')
            self.assertEqual(p.dumpPython(None),"cms.Path(process.a*process.b)\n")
            p2 = Path((b+a))
            #print p2.dumpConfig('')
            self.assertEqual(p2.dumpPython(None),"cms.Path(process.b+process.a)\n")
            c = DummyModule('c')
            p3 = Path(c*(a+b))
            #print p3.dumpConfig('')
            self.assertEqual(p3.dumpPython(None),"cms.Path(process.c*(process.a+process.b))\n")
            p4 = Path(c*a+b)
            #print p4.dumpConfig('')
            self.assertEqual(p4.dumpPython(None),"cms.Path(process.c*process.a+process.b)\n")
            p5 = Path(a+ignore(b))
            #print p5.dumpConfig('')
            self.assertEqual(p5.dumpPython(None),"cms.Path(process.a+cms.ignore(process.b))\n")
            p6 = Path(c+a*b)
            #print p6.dumpConfig('')
            self.assertEqual(p6.dumpPython(None),"cms.Path(process.c+process.a*process.b)\n")
            p7 = Path(a+~b)
            self.assertEqual(p7.dumpPython(None),"cms.Path(process.a+~process.b)\n")
            p8 = Path((a+b)*c)
            self.assertEqual(p8.dumpPython(None),"cms.Path((process.a+process.b)*process.c)\n")
            l = list()
            namesVisitor = DecoratedNodeNameVisitor(l)
            p.visit(namesVisitor)
            self.assertEqual(l, ['a', 'b'])
            l[:] = []
            p5.visit(namesVisitor)
            self.assertEqual(l, ['a', '-b'])
            l[:] = []
            p7.visit(namesVisitor)
            self.assertEqual(l, ['a', '!b'])
            l[:] = []
            moduleVisitor = ModuleNodeVisitor(l)
            p8.visit(moduleVisitor)
            names = [m.label_() for m in l]
            self.assertEqual(names, ['a', 'b', 'c'])

        def testVisitor(self):
            class TestVisitor(object):
                def __init__(self, enters, leaves):
                    self._enters = enters
                    self._leaves = leaves
                def enter(self,visitee):
                    #print visitee
                    if self._enters[0] != visitee:
                        raise RuntimeError("wrong node ("+str(visitee)+") on 'enter'")
                    else:
                        self._enters = self._enters[1:]
                def leave(self,visitee):
                    if self._leaves[0] != visitee:
                        raise RuntimeError("wrong node ("+str(visitee)+") on 'leave'\n expected ("+str(self._leaves[0])+")")
                    else:
                        self._leaves = self._leaves[1:]
            a = DummyModule("a")
            b = DummyModule('b')
            multAB = a*b
            p = Path(multAB)
            t = TestVisitor(enters=[multAB,a,b],
                            leaves=[a,b,multAB])
            p.visit(t)

            plusAB = a+b
            p = Path(plusAB)
            t = TestVisitor(enters=[plusAB,a,b],
                            leaves=[a,b,plusAB])
            p.visit(t)
            
            s=Sequence(plusAB)
            c=DummyModule("c")
            multSC = s*c
            p=Path(multSC)
            t=TestVisitor(enters=[multSC,s,plusAB,a,b,c],
                          leaves=[a,b,plusAB,s,c,multSC])
            p.visit(t)
            
            notA= ~a
            p=Path(notA)
            t=TestVisitor(enters=[notA,a],leaves=[a,notA])
            p.visit(t)
        def testResolve(self):
            m1 = DummyModule("m1")
            m2 = DummyModule("m2")
            s1 = Sequence(m1)
            s2 = SequencePlaceholder("s3")
            s3 = Sequence(m2)
            p = Path(s1*s2)
            l = list()
            d = dict()
            d['s1'] = s1
            d['s2'] = s2
            d['s3'] = s3
            #resolver = ResolveVisitor(d)
            #p.visit(resolver)
            namesVisitor = DecoratedNodeNameVisitor(l)
            p.visit(namesVisitor)
            self.assertEqual(l, ['m1'])
            p.resolve(d)
            l[:] = []
            p.visit(namesVisitor)
            self.assertEqual(l, ['m1', 'm2'])
        def testReplace(self):
            m1 = DummyModule("m1")
            m2 = DummyModule("m2")
            m3 = DummyModule("m3")
            m4 = DummyModule("m4")
            m5 = DummyModule("m5")
 
            s1 = Sequence(m1*~m2*m1*m2*ignore(m2))
            s2 = Sequence(m1*m2)
            l = []
            namesVisitor = DecoratedNodeNameVisitor(l)
            s1.visit(namesVisitor)
            self.assertEqual(l,['m1', '!m2', 'm1', 'm2', '-m2'])

            s3 = Sequence(~m1*s2)
            s3.replace(~m1, m2)
            l[:] = []
            s3.visit(namesVisitor)
            self.assertEqual(l, ['m2', 'm1', 'm2'])

            s1.replace(m2,m3)
            l[:] = []
            s1.visit(namesVisitor)
            self.assertEqual(l,['m1', '!m3', 'm1', 'm3', '-m3'])
            s2 = Sequence(m1*m2)
            s3 = Sequence(~m1*s2)
            l[:] = []
            s3.visit(namesVisitor)
            self.assertEqual(l,['!m1', 'm1', 'm2'])
            l[:] = []
            s3.replace(s2,m1)
            s3.visit(namesVisitor)
            self.assertEqual(l,['!m1', 'm1'])

        def testExpandAndClone(self):
            m1 = DummyModule("m1")
            m2 = DummyModule("m2")
            m3 = DummyModule("m3")
            m4 = DummyModule("m4")
            m5 = DummyModule("m5")

            s1 = Sequence(m1*~m2*m1*m2*ignore(m2))
            s2 = Sequence(m1*m2)
            s3 = Sequence(~m1*s2)

            p = Path(s1+s3)
            p2 = p.expandAndClone()
            l = []
            namesVisitor = DecoratedNodeNameVisitor(l)
            p2.visit(namesVisitor)
            self.assertEqual(l, ['m1', '!m2', 'm1', 'm2', '-m2', '!m1', 'm1', 'm2'])

        def testAdd(self):
            m1 = DummyModule("m1")
            m2 = DummyModule("m2")
            m3 = DummyModule("m3")
            m4 = DummyModule("m4")
            s1 = Sequence(m1)
            s3 = Sequence(m3+ignore(m4))
            p = Path(s1)
            p += ~m2
            p *= s3

            l = []
            namesVisitor = DecoratedNodeNameVisitor(l)
            p.visit(namesVisitor)
            self.assertEqual(l, ['m1', '!m2', 'm3', '-m4'])
            
            s4 = Sequence()
            s4 +=m1
            l[:]=[]; s1.visit(namesVisitor); self.assertEqual(l,['m1'])
            self.assertEqual(s4.dumpPython(None),"cms.Sequence(process.m1)\n")
            s4 = Sequence()
            s4 *=m1
            l[:]=[]; s1.visit(namesVisitor); self.assertEqual(l,['m1'])
            self.assertEqual(s4.dumpPython(None),"cms.Sequence(process.m1)\n")


        def testRemove(self):
            m1 = DummyModule("m1")
            m2 = DummyModule("m2")
            m3 = DummyModule("m3")
            s1 = Sequence(m1*m2+~m3)
            s2 = Sequence(m1*s1)
            l = []
            namesVisitor = DecoratedNodeNameVisitor(l)
            d = {'m1':m1 ,'m2':m2, 'm3':m3,'s1':s1, 's2':s2}  
            l[:] = []; s1.visit(namesVisitor); self.assertEqual(l,['m1', 'm2', '!m3'])
            l[:] = []; s2.visit(namesVisitor); self.assertEqual(l,['m1', 'm1', 'm2', '!m3'])
            s1.remove(m2)
            l[:] = []; s1.visit(namesVisitor); self.assertEqual(l,['m1', '!m3'])
            l[:] = []; s2.visit(namesVisitor); self.assertEqual(l,['m1', 'm1', '!m3'])
            s2.remove(m3)
            l[:] = []; s1.visit(namesVisitor); self.assertEqual(l,['m1'])
            l[:] = []; s2.visit(namesVisitor); self.assertEqual(l,['m1', 'm1'])
            s1 = Sequence( m1 + m2 + m1 + m2 )
            l[:] = []; s1.visit(namesVisitor); self.assertEqual(l,['m1', 'm2', 'm1', 'm2'])
            s1.remove(m2) 
            l[:] = []; s1.visit(namesVisitor); self.assertEqual(l,['m1', 'm1', 'm2'])
            s1 = Sequence( m1 + m3 )
            s2 = Sequence( m2 + ignore(m3) + s1 + m3 )
            l[:] = []; s2.visit(namesVisitor); self.assertEqual(l,['m2', '-m3', 'm1', 'm3', 'm3'])
            s2.remove(s1)
            l[:] = []; s2.visit(namesVisitor); self.assertEqual(l,['m2', '-m3', 'm3'])
            s2.remove(m3)
            l[:] = []; s2.visit(namesVisitor); self.assertEqual(l,['m2', 'm3'])
            s1 = Sequence(m1*m2*m3)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1*process.m2*process.m3)\n")
            s1.remove(m2)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1*process.m3)\n")
            s1 = Sequence(m1+m2+m3)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m2+process.m3)\n")
            s1.remove(m2)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m3)\n")
            s1 = Sequence(m1*m2+m3)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1*process.m2+process.m3)\n")
            s1.remove(m2)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m3)\n")
            s1 = Sequence(m1+m2*m3)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m2*process.m3)\n")
            s1.remove(m2)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m3)\n")
            s1.remove(m1)
            s1.remove(m3)
            l[:]=[]; s1.visit(namesVisitor); self.assertEqual(l,[])
            self.assertEqual(s1.dumpPython(None), "cms.Sequence()\n")
            s3 = Sequence(m1)
            s3.remove(m1)
            l[:]=[]; s3.visit(namesVisitor); self.assertEqual(l,[])
            self.assertEqual(s3.dumpPython(None), "cms.Sequence()\n")
            s3 = Sequence(m1)
            s4 = Sequence(s3)
            s4.remove(m1)
            l[:]=[]; s4.visit(namesVisitor); self.assertEqual(l,[])
            self.assertEqual(s4.dumpPython(None), "cms.Sequence()\n")
            
        def testCopyAndExclude(self):
            a = DummyModule("a")
            b = DummyModule("b")
            c = DummyModule("c")
            d = DummyModule("d")
            s = Sequence(a+b+c)
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a+process.b+process.c)\n")
            s = Sequence(a+b+c+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a+process.b+process.c)\n")
            s=Sequence(a*b+c+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a*process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a*process.b+process.c)\n")
            s = Sequence(a+b*c+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b*process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a+process.b*process.c)\n")
            s2 = Sequence(a+b)
            s = Sequence(c+s2+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.c+process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.c+process.a+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence((process.a+process.b)+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.c+(process.a+process.b))\n")
            self.assertEqual(s.copyAndExclude([a,b]).dumpPython(None),"cms.Sequence(process.c+process.d)\n")
            s2 = Sequence(a+b+c)
            s = Sequence(s2+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence((process.a+process.b+process.c))\n")
            s2 = Sequence(a+b+c)
            s = Sequence(s2*d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence((process.b+process.c)*process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence((process.a+process.c)*process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence((process.a+process.b)*process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence((process.a+process.b+process.c))\n")
            self.assertEqual(s.copyAndExclude([a,b,c]).dumpPython(None),"cms.Sequence(process.d)\n")
            s = Sequence(ignore(a)+b+c+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(cms.ignore(process.a)+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(cms.ignore(process.a)+process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(cms.ignore(process.a)+process.b+process.c)\n")
            s = Sequence(a+ignore(b)+c+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(cms.ignore(process.b)+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+cms.ignore(process.b)+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a+cms.ignore(process.b)+process.c)\n")
            s = Sequence(a+b+c+ignore(d))
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+cms.ignore(process.d))\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+cms.ignore(process.d))\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+process.b+cms.ignore(process.d))\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a+process.b+process.c)\n")
            s = Sequence(~a+b+c+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(~process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(~process.a+process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(~process.a+process.b+process.c)\n")
            s = Sequence(a+~b+c+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(~process.b+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+~process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a+~process.b+process.c)\n")
            s = Sequence(a+b+c+~d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+~process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+~process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+process.b+~process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a+process.b+process.c)\n")


            
        def testDependencies(self):
            m1 = DummyModule("m1")
            m2 = DummyModule("m2")
            m3 = DummyModule("m3")
            m4 = DummyModule("m4")
            m5 = DummyModule("m5")
            deps = dict()
            s4 = Sequence(~m1*(m2+m3)+m4)
            s4.setLabel('s4')
            s4.findHardDependencies('top',deps)
            self.assertEqual(deps['m2'][0].depSet, set(['m1']))
            self.assertEqual(deps['m3'][0].depSet, set(['m1']))
            self.assertEqual(deps['m2'][0].sequenceName, 's4')
            self.assertEqual(deps['m3'][0].sequenceName, 's4')
            self.failIf(deps.has_key('m4'))
            self.failIf(deps.has_key('m1'))
            deps = dict()
            p5 = Path(s4*m5)
            p5.setLabel('p5')
            p5.findHardDependencies('top',deps)
            self.assertEqual(len(deps['m5'][0].depSet), 4)
            self.assertEqual(deps['m5'][0].sequenceName, 'p5')
            self.assertEqual(deps['m3'][0].sequenceName, 's4')
        def testSequenceTypeChecks(self):
            m1 = DummyModule("m1")
            m2 = DummyModule("m2")
            s1 = Sequence(m1*m2)
            def testRaise():
                s1.something = 1
            self.assertRaises(AttributeError,testRaise)
            def testRaise2():
                s2 = Sequence(m1*None)
            self.assertRaises(TypeError,testRaise2)


                        
    unittest.main()
                          


                           
    

        
        
