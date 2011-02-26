
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
        return _SequenceCollection(self,rhs)
    def __add__(self,rhs):
        return _SequenceCollection(self,rhs)
    def __invert__(self):
        return _SequenceNegation(self)
    def _clonesequence(self, lookuptable):
        try: 
            return lookuptable[id(self)]
        except:
            raise KeyError("no "+str(type(self))+" with id "+str(id(self))+" found")
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
    def _appendToCollection(self,collection):
        collection.append(self)
        
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

class _SequenceCollection(_Sequenceable):
    """Holds representation of the operations without having to use recursion.
    Operations are added to the beginning of the list and their operands are
    added to the end of the list, with the left added before the right
    """
    def __init__(self,*seqList):
        self._collection = list()
        for s in seqList:
            _checkIfSequenceable(self,s)
            s._appendToCollection(self._collection)
    def __mul__(self,rhs):
        _checkIfSequenceable(self,rhs)
        rhs._appendToCollection(self._collection)
        return self
    def __add__(self,rhs):
        _checkIfSequenceable(self,rhs)
        rhs._appendToCollection(self._collection)
        return self
    def __str__(self):
        sep = ''
        returnValue = ''
        for m in self._collection:
            if m is not None:
                returnValue += sep+str(m)
                sep = '+'
        return returnValue
    def _appendToCollection(self,collection):
        collection.extend(self._collection)
    def dumpSequencePython(self):
        returnValue = self._collection[0].dumpSequencePython()
        for m in self._collection[1:]:
            returnValue += '+'+m.dumpSequencePython()        
        return returnValue
    def dumpSequenceConfig(self):
        returnValue = self._collection[0].dumpSequenceConfig()
        for m in self._collection[1:]:
            returnValue += '&'+m.dumpSequenceConfig()        
        return returnValue
    def visitNode(self,visitor):
        for m in self._collection:
            m.visitNode(visitor)
    def resolve(self, processDict):
        self._collection = [x.resolve(processDict) for x in self._collection]
        return self
    def index(self,item):
        return self._collection.index(item)
    def insert(self,index,item):
        self._collection.insert(index,item)



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
            self._seq = _SequenceCollection()
            arg[0]._appendToCollection(self._seq._collection)
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
            self.__dict__["_seq"] = _SequenceCollection()
        self._seq+=rhs
        return self
    def __iadd__(self,rhs):
        _checkIfSequenceable(self, rhs)
        if self._seq is None:
            self.__dict__["_seq"] = _SequenceCollection()
        self._seq += rhs
        return self
    def __str__(self):
        return str(self._seq)
    def dumpConfig(self, options):
        s = ''
        if self._seq is not None:
            s = self._seq.dumpSequenceConfig()
        return '{'+s+'}\n'
    def dumpPython(self, options):
        """Returns a string which is the python representation of the object"""
        s=''
        if self._seq is not None:
            s =self._seq.dumpSequencePython()
        return 'cms.'+type(self).__name__+'('+s+')\n'
    def dumpSequencePython(self):
        """Returns a string which contains the python representation of just the internal sequence"""
        # only dump the label, if possible
        if self.hasLabel_():
            return _Labelable.dumpSequencePython(self)
        else:
            # dump it verbose
            if self._seq is None:
                return ''
            return '('+self._seq.dumpSequencePython()+')'
    def dumpSequenceConfig(self):
        """Returns a string which contains the old config language representation of just the internal sequence"""
        # only dump the label, if possible
        if self.hasLabel_():
            return _Labelable.dumpSequenceConfig(self)
        else:
            # dump it verbose
            if self._seq is None:
                return ''
            return '('+self._seq.dumpSequenceConfig()+')'
    def __repr__(self):
        s = ''
        if self._seq is not None:
           s = str(self._seq)
        return "cms."+type(self).__name__+'('+s+')\n'
    def moduleNames(self):
        """Returns a set containing the names of all modules being used"""
        result = set()
        visitor = NodeNameVisitor(result)
        self.visit(visitor)
        return result
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
        """Finds all instances of 'original' and substitutes 'replacement' for them.
           Returns 'True' if a replacement occurs."""
        if not isinstance(original,_Sequenceable) or not isinstance(replacement,_Sequenceable):
           raise TypeError("replace only works with sequenceable objects")
        else:
            v = _CopyAndReplaceSequenceVisitor(original,replacement)
            self.visit(v)
            if v.didReplace():
                self._seq = v.result()
            return v.didReplace()
    def index(self,item):
        """Returns the index at which the item is found or raises an exception"""
        if self._seq is not None:
            return self._seq.index(item)
        raise ValueError(str(item)+" is not in the sequence")
    def insert(self,index,item):
        """Inserts the item at the index specified"""
        _checkIfSequenceable(self, item)
        self._seq.insert(index,item)
    def remove(self, something):
        """Remove the first occurrence of 'something' (a sequence or a module)
           Returns 'True' if the module has been removed, False if it was not found"""
        v = _CopyAndRemoveFirstSequenceVisitor(something)
        self.visit(v)
        if v.didRemove():
            self._seq = v.result()
        return v.didRemove()
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
            #print str(processDict.keys())
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
                #print str(self.processDict.keys())
                raise RuntimeError("The SequencePlaceholder "+visitee._name+ " cannot be resolved.\n Known keys are:"+str(self.processDict.keys()))
            visitee = self.processDict[visitee._name]
    def leave(self,visitee):
       if isinstance(visitee, SequencePlaceholder):
           pass


class _CopyAndExcludeSequenceVisitorOld(object):
   """Traverses a Sequence and constructs a new sequence which does not contain modules from the specified list"""
   def __init__(self,modulesToRemove):
       self.__modulesToIgnore = modulesToRemove
       self.__stack = list()
       self.__stack.append(list())
       self.__result = None
       self.__didExclude = False
   def enter(self,visitee):
       if len(self.__stack) > 0:
           #add visitee to its parent's stack entry
           self.__stack[-1].append([visitee,False])
       if isinstance(visitee,_SequenceLeaf):
           if visitee in self.__modulesToIgnore:
               self.__didExclude = True
               self.__stack[-1][-1]=[None,True]
       elif isinstance(visitee, Sequence):
           if visitee in self.__modulesToIgnore:
               self.__didExclude = True
               self.__stack[-1][-1]=[None,True]
           self.__stack.append(list())
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
               self.__didExclude = True
               if countNulls != 0:
                   #this node must go away
                   if len(nonNulls) == 0:
                       #all subnodes went away 
                       node = None
                   else:
                       node = nonNulls[0]
                       for n in nonNulls[1:]:
                           node = node+n
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
       if not isinstance(visitee,_SequenceLeaf):
           self.__stack = self.__stack[:-1]        
   def result(self):
       result = None
       for n in (x[0] for x in self.__stack[0]):
           if n is None:
               continue
           if result is None:
               result = n
           else:
               result = result+n
       return result
   def didExclude(self):
       return self.__didExclude


class _MutatingSequenceVisitor(object):
    """Traverses a Sequence and constructs a new sequence by applying the operator to each element of the sequence"""
    def __init__(self,operator):
      self.__operator = operator
      self.__stack = list()
      self.__stack.append(list())
      self.__result = None
      self.__didApply = False
    def enter(self,visitee):
      if len(self.__stack) > 0:
          #add visitee to its parent's stack entry
          self.__stack[-1].append([visitee,False])
      v = self.__operator(visitee)
      if v is not visitee:
          #was changed
          self.__didApply = True
          self.__stack[-1][-1]=[v,True]
      if not isinstance(visitee, _SequenceLeaf):
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
                      node = nonNulls[0]
                      for n in nonNulls[1:]:
                          node = node+n
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
                      for n in nonNulls[1:]:
                          node = node+n

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
      if not isinstance(visitee,_SequenceLeaf):
          self.__stack = self.__stack[:-1]        
    def result(self):
      result = None
      for n in (x[0] for x in self.__stack[0]):
          if n is None:
              continue
          if result is None:
              result = n
          else:
              result = result+n
      return result
    def _didApply(self):
      return self.__didApply

class _CopyAndRemoveFirstSequenceVisitor(_MutatingSequenceVisitor):
    """Traverses a Sequence and constructs a new sequence which does not contain modules from the specified list"""
    def __init__(self,moduleToRemove):
        class _RemoveFirstOperator(object):
            def __init__(self,moduleToRemove):
                self.__moduleToRemove = moduleToRemove
                self.__found = False
            def __call__(self,test):
                if not self.__found and test is self.__moduleToRemove:
                    self.__found = True
                    return None
                return test
        super(type(self),self).__init__(_RemoveFirstOperator(moduleToRemove))
    def didRemove(self):
        return self._didApply()

class _CopyAndExcludeSequenceVisitor(_MutatingSequenceVisitor):
    """Traverses a Sequence and constructs a new sequence which does not contain the module specified"""
    def __init__(self,modulesToRemove):
        class _ExcludeOperator(object):
            def __init__(self,modulesToRemove):
                self.__modulesToIgnore = modulesToRemove
            def __call__(self,test):
                if test in modulesToRemove:
                    return None
                return test
        super(type(self),self).__init__(_ExcludeOperator(modulesToRemove))
    def didExclude(self):
        return self._didApply()

class _CopyAndReplaceSequenceVisitor(_MutatingSequenceVisitor):
    """Traverses a Sequence and constructs a new sequence which  replaces a specified module with a different module"""
    def __init__(self,target,replace):
        class _ReplaceOperator(object):
            def __init__(self,target,replace):
                self.__target = target
                self.__replace = replace
            def __call__(self,test):
                if test == self.__target:
                    return self.__replace
                return test
        super(type(self),self).__init__(_ReplaceOperator(target,replace))
    def didReplace(self):
        return self._didApply()



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
            self.assertEqual(p.dumpPython(None),"cms.Path(process.a+process.b)\n")
            p2 = Path((b+a))
            #print p2.dumpConfig('')
            self.assertEqual(p2.dumpPython(None),"cms.Path(process.b+process.a)\n")
            c = DummyModule('c')
            p3 = Path(c*(a+b))
            #print p3.dumpConfig('')
            self.assertEqual(p3.dumpPython(None),"cms.Path(process.c+process.a+process.b)\n")
            p4 = Path(c*a+b)
            #print p4.dumpConfig('')
            self.assertEqual(p4.dumpPython(None),"cms.Path(process.c+process.a+process.b)\n")
            p5 = Path(a+ignore(b))
            #print p5.dumpConfig('')
            self.assertEqual(p5.dumpPython(None),"cms.Path(process.a+cms.ignore(process.b))\n")
            p6 = Path(c+a*b)
            #print p6.dumpConfig('')
            self.assertEqual(p6.dumpPython(None),"cms.Path(process.c+process.a+process.b)\n")
            p7 = Path(a+~b)
            self.assertEqual(p7.dumpPython(None),"cms.Path(process.a+~process.b)\n")
            p8 = Path((a+b)*c)
            self.assertEqual(p8.dumpPython(None),"cms.Path(process.a+process.b+process.c)\n")
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
        def testDumpConfig(self):
            a = DummyModule("a")
            b = DummyModule('b')
            p = Path((a*b))
            #print p.dumpConfig('')
            self.assertEqual(p.dumpConfig(None),"{a&b}\n")
            p2 = Path((b+a))
            #print p2.dumpConfig('')
            self.assertEqual(p2.dumpConfig(None),"{b&a}\n")
            c = DummyModule('c')
            p3 = Path(c*(a+b))
            #print p3.dumpConfig('')
            self.assertEqual(p3.dumpConfig(None),"{c&a&b}\n")
            p4 = Path(c*a+b)
            #print p4.dumpConfig('')
            self.assertEqual(p4.dumpConfig(None),"{c&a&b}\n")
            p5 = Path(a+ignore(b))
            #print p5.dumpConfig('')
            self.assertEqual(p5.dumpConfig(None),"{a&-b}\n")
            p6 = Path(c+a*b)
            #print p6.dumpConfig('')
            self.assertEqual(p6.dumpConfig(None),"{c&a&b}\n")
            p7 = Path(a+~b)
            self.assertEqual(p7.dumpConfig(None),"{a&!b}\n")
            p8 = Path((a+b)*c)
            self.assertEqual(p8.dumpConfig(None),"{a&b&c}\n")        
        def testVisitor(self):
            class TestVisitor(object):
                def __init__(self, enters, leaves):
                    self._enters = enters
                    self._leaves = leaves
                def enter(self,visitee):
                    #print visitee.dumpSequencePython()
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
            t = TestVisitor(enters=[a,b],
                            leaves=[a,b])
            p.visit(t)

            plusAB = a+b
            p = Path(plusAB)
            t = TestVisitor(enters=[a,b],
                            leaves=[a,b])
            p.visit(t)
            
            s=Sequence(plusAB)
            c=DummyModule("c")
            multSC = s*c
            p=Path(multSC)
            t=TestVisitor(enters=[s,a,b,c],
                          leaves=[a,b,s,c])
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
            
            s1 = Sequence(m1+m2)
            s2 = Sequence(m3+m4)
            s3 = Sequence(s1+s2)
            s3.replace(m3,m5)
            l[:] = []
            s3.visit(namesVisitor)
            self.assertEqual(l,['m1','m2','m5','m4'])
        def testIndex(self):
            m1 = DummyModule("a")
            m2 = DummyModule("b")
            m3 = DummyModule("c")
        
            s = Sequence(m1+m2+m3)
            self.assertEqual(s.index(m1),0)
            self.assertEqual(s.index(m2),1)        
            self.assertEqual(s.index(m3),2)

        def testInsert(self):
            m1 = DummyModule("a")
            m2 = DummyModule("b")
            m3 = DummyModule("c")
            s = Sequence(m1+m3)
            s.insert(1,m2)
            self.assertEqual(s.index(m1),0)
            self.assertEqual(s.index(m2),1)        
            self.assertEqual(s.index(m3),2)
            
        
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
            print s1._seq.dumpSequencePython()
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
            l[:] = []; s1.visit(namesVisitor); self.assertEqual(l,['m1', '!m3'])
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
            l[:] = []; s2.visit(namesVisitor); self.assertEqual(l,['m2','m3'])
            s1 = Sequence(m1*m2*m3)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m2+process.m3)\n")
            s1.remove(m2)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m3)\n")
            s1 = Sequence(m1+m2+m3)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m2+process.m3)\n")
            s1.remove(m2)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m3)\n")
            s1 = Sequence(m1*m2+m3)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m2+process.m3)\n")
            s1.remove(m2)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m3)\n")
            s1 = Sequence(m1+m2*m3)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m2+process.m3)\n")
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
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a+process.b+process.c)\n")
            s = Sequence(a+b*c+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a+process.b+process.c)\n")
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
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+process.b+process.d)\n")
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
        def testCopy(self):
            a = DummyModule("a")
            b = DummyModule("b")
            c = DummyModule("c")
            p1 = Path(a+b+c)
            p2 = p1.copy()
            e = DummyModule("e")
            p2.replace(b,e)
            self.assertEqual(p1.dumpPython(None),"cms.Path(process.a+process.b+process.c)\n")
            self.assertEqual(p2.dumpPython(None),"cms.Path(process.a+process.e+process.c)\n")
        def testInsertInto(self):
            from FWCore.ParameterSet.Types import vstring
            class TestPSet(object):
                def __init__(self):
                    self._dict = dict()
                def addVString(self,isTracked,label,value):
                    self._dict[label]=value
            a = DummyModule("a")
            b = DummyModule("b")
            c = DummyModule("c")
            d = DummyModule("d")
            p = Path(a+b+c+d)
            ps = TestPSet()
            p.insertInto(ps,"p",dict())
            self.assertEqual(ps._dict, {"p":vstring("a","b","c","d")})
            s = Sequence(b+c)
            p = Path(a+s+d)
            ps = TestPSet()
            p.insertInto(ps,"p",dict())
            self.assertEqual(ps._dict, {"p":vstring("a","b","c","d")})
                        
    unittest.main()
                          


                           
    

        
        
