
from Mixins import _ConfigureComponent, PrintOptions
from Mixins import _Labelable, _Unlabelable
from Mixins import _ValidatingParameterListBase
from ExceptionHandling import *
from OrderedSet import OrderedSet

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
    def resolve(self, processDict,keepIfCannotResolve=False):
        return self
    def isOperation(self):
        """Returns True if the object is an operator (e.g. *,+ or !) type"""
        return False
    def isLeaf(self):
        return False
    def _visitSubNodes(self,visitor):
        pass
    def visitNode(self,visitor):
        visitor.enter(self)
        self._visitSubNodes(visitor)
        visitor.leave(self)
    def _appendToCollection(self,collection):
        collection.append(self)
    def _errorstr(self):
        return "A Sequenceable type"

def _checkIfSequenceable(caller, v):
    if not isinstance(v,_Sequenceable):
        typename = format_typename(caller)
        msg = format_outerframe(2)
        msg += "%s only takes arguments of types which are allowed in a sequence, but was given:\n" %typename
        msg +=format_typename(v)
        msg +="\nPlease remove the problematic object from the argument list"
        raise TypeError(msg)

def _checkIfBooleanLogicSequenceable(caller, v):
    if not isinstance(v,_BooleanLogicSequenceable):
        typename = format_typename(caller)
        msg = format_outerframe(2)
        msg += "%s only takes arguments of types which are allowed in a boolean logic sequence, but was given:\n" %typename
        msg +=format_typename(v)
        msg +="\nPlease remove the problematic object from the argument list"
        raise TypeError(msg)

class _BooleanLogicSequenceable(_Sequenceable):
    """Denotes an object which can be used in a boolean logic sequence"""
    def __init__(self):
        super(_BooleanLogicSequenceable,self).__init__()
    def __or__(self,other):
        return _BooleanLogicExpression(_BooleanLogicExpression.OR,self,other)
    def __and__(self,other):
        return _BooleanLogicExpression(_BooleanLogicExpression.AND,self,other)


class _BooleanLogicExpression(_BooleanLogicSequenceable):
    """Contains the operation of a boolean logic expression"""
    OR = 0
    AND = 1
    def __init__(self,op,left,right):
        _checkIfBooleanLogicSequenceable(self,left)
        _checkIfBooleanLogicSequenceable(self,right)
        self._op = op
        self._items = list()
        #if either the left or right side are the same kind of boolean expression
        # then we can just add their items to our own. This keeps the expression
        # tree more compact
        if isinstance(left,_BooleanLogicExpression) and left._op == self._op:
            self._items.extend(left._items)
        else:
            self._items.append(left)
        if isinstance(right,_BooleanLogicExpression) and right._op == self._op:
            self._items.extend(right._items)
        else:
            self._items.append(right)
    def isOperation(self):
        return True
    def _visitSubNodes(self,visitor):
        for i in self._items:
            i.visitNode(visitor)
    def dumpSequencePython(self, options=PrintOptions()):
        returnValue = ''
        join = ''
        operatorJoin =self.operatorString()
        for m in self._items:
            returnValue +=join
            join = operatorJoin
            if not isinstance(m,_BooleanLogicSequenceLeaf):
                returnValue += '('+m.dumpSequencePython(options)+')'
            else:
                returnValue += m.dumpSequencePython(options)
        return returnValue
    def operatorString(self):
        returnValue ='|'
        if self._op == self.AND:
            returnValue = '&'
        return returnValue


class _SequenceLeaf(_Sequenceable):
    def __init__(self):
        pass
    def isLeaf(self):
        return True


class _BooleanLogicSequenceLeaf(_BooleanLogicSequenceable):
    def __init__(self):
        pass
    def isLeaf(self):
        return True

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
    def dumpSequencePython(self, options=PrintOptions()):
        returnValue = ''
        separator = ''
        for item in self._collection:
            itemDump = item.dumpSequencePython(options)
            if itemDump:
                returnValue += (separator + itemDump)
                separator = '+'
        return returnValue
    def dumpSequenceConfig(self):
        returnValue = self._collection[0].dumpSequenceConfig()
        for m in self._collection[1:]:
            returnValue += '&'+m.dumpSequenceConfig()        
        return returnValue
    def visitNode(self,visitor):
        for m in self._collection:
            m.visitNode(visitor)
    def resolve(self, processDict,keepIfCannotResolve=False):
        self._collection = [x.resolve(processDict,keepIfCannotResolve) for x in self._collection]
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
        if (len(arg) > 1 and not isinstance(arg[1], Task)) or (len(arg) > 0 and not isinstance(arg[0],_Sequenceable) and not isinstance(arg[0],Task)):
            typename = format_typename(self)
            msg = format_outerframe(2) 
            msg += "The %s constructor takes zero or one sequenceable argument followed by zero or more arguments of type Task. But the following types are given:\n" %typename
            for item,i in zip(arg, xrange(1,20)):
                try:
                    msg += "    %i) %s \n"  %(i, item._errorstr())
                except:
                    msg += "    %i) Not sequenceable and not a Task\n" %(i)
            if len(arg) > 1 and isinstance(arg[0],_Sequenceable) and isinstance(arg[1], _Sequenceable):
                msg += "Maybe you forgot to combine the sequenceable arguments via '*' or '+'."
            raise TypeError(msg)
        tasks = arg
        if len(arg) > 0 and isinstance(arg[0], _Sequenceable):
            self._seq = _SequenceCollection()
            arg[0]._appendToCollection(self._seq._collection)
            tasks = arg[1:]
        self._isModified = False

        self._tasks = OrderedSet()

        if len(tasks) > 0:
            self.associate(*tasks)
    def associate(self,*tasks):
        for task in tasks:
            if not isinstance(task, Task):
                raise TypeError("associate only works with objects of type Task")
            self._tasks.add(task)
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
        v = ExpandVisitor(type(self))
        self.visit(v)
        return v.resultString()
    def dumpConfig(self, options):
        s = ''
        if self._seq is not None:
            s = self._seq.dumpSequenceConfig()
        return '{'+s+'}\n'
    def dumpPython(self, options=PrintOptions()):
        """Returns a string which is the python representation of the object"""
        s = self.dumpPythonNoNewline(options)
        return s + "\n"
    def dumpPythonNoNewline(self, options=PrintOptions()):
        s=''
        if self._seq is not None:
            s =self._seq.dumpSequencePython(options)
        associationContents = set()
        for task in self._tasks:
            if task.hasLabel_():
                associationContents.add(_Labelable.dumpSequencePython(task, options))
            else:
                associationContents.add(task.dumpPythonNoNewline(options))
        for iString in sorted(associationContents):
            if s:
                s += ", "
            s += iString
        if len(associationContents) > 254:
            return 'cms.'+type(self).__name__+'(*['+s+'])'
        return 'cms.'+type(self).__name__+'('+s+')'
    def dumpSequencePython(self, options=PrintOptions()):
        """Returns a string which contains the python representation of just the internal sequence"""
        # only dump the label, if possible
        if self.hasLabel_():
            return _Labelable.dumpSequencePython(self, options)
        elif len(self._tasks) == 0:
            if self._seq is None:
                return ''
            s = self._seq.dumpSequencePython(options)
            if s:
              return '('+s+')'
            return ''
        return self.dumpPythonNoNewline(options)
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
    def contains(self, mod):
        visitor = ContainsModuleVisitor(mod)
        self.visit(visitor)
        return visitor.result()
    def copy(self):
        returnValue =_ModuleSequenceType.__new__(type(self))
        if self._seq is not None:
            returnValue.__init__(self._seq)
        else:
            returnValue.__init__()
        returnValue._tasks = OrderedSet(self._tasks)
        return returnValue
    def copyAndExclude(self,listOfModulesToExclude):
        """Returns a copy of the sequence which excludes those module in 'listOfModulesToExclude'"""
        # You can exclude instances of these types EDProducer, EDFilter, OutputModule,
        # EDAnalyzer, ESSource, ESProducer, Service, Sequence, SequencePlaceholder, Task,
        # _SequenceNegation, and _SequenceIgnore.
        # Mostly this is very intuitive, but there are some complications in cases
        # where objects that contain other objects are involved. See the comments
        # for the _MutatingSequenceVisitor.
        v = _CopyAndExcludeSequenceVisitor(listOfModulesToExclude)
        self.visit(v)
        result = self.__new__(type(self))
        result.__init__(v.result(self)[0], *v.result(self)[1])
        return result
    def expandAndClone(self):
        # Name of this function is not very good. It makes a shallow copy with all
        # the subTasks and subSequences flattened out (removed), but keeping all the
        # modules that were in those subSequences and subTasks as well as the top level
        # ones. Note this will also remove placeholders so one should probably
        # call resolve before using this if the sequence contains any placeholders.
        visitor = ExpandVisitor(type(self))
        self.visit(visitor)
        return visitor.result()
    def _postProcessFixup(self,lookuptable):
        self._seq = self._seq._clonesequence(lookuptable)
        return self
    def replace(self, original, replacement):
        """Finds all instances of 'original' and substitutes 'replacement' for them.
           Returns 'True' if a replacement occurs."""
        # This works for either argument being of type EDProducer, EDFilter, OutputModule,
        # EDAnalyzer, ESProducer, ESSource, Service, Sequence, SequencePlaceHolder,
        # Task, _SequenceNegation, _SequenceIgnore. Although it will fail with a
        # raised exception if the replacement actually hits a case where a
        # non-Sequenceable object is placed in the sequenced part of a Sequence
        # or a type not allowed on a Task is put on a Task.
        # There is one special case where we need an explicit check to prevent
        # the algorithm from getting confused, either both or neither can be Tasks
        #
        # Mostly this is very intuitive, but there are some complications in cases
        # where objects that contain other objects are involved. See the comments
        # for the _MutatingSequenceVisitor.

        if isinstance(original,Task) != isinstance(replacement,Task):
               raise TypeError("replace only works if both arguments are Tasks or neither")
        v = _CopyAndReplaceSequenceVisitor(original,replacement)
        self.visit(v)
        if v.didReplace():
            self._seq = v.result(self)[0]
            self._tasks.clear()
            self.associate(*v.result(self)[1])
        return v.didReplace()
    def index(self,item):
        """Returns the index at which the item is found or raises an exception"""
        if self._seq is not None:
            return self._seq.index(item)
        raise ValueError(str(item)+" is not in the sequence")
    def insert(self,index,item):
        """Inserts the item at the index specified"""
        _checkIfSequenceable(self, item)
        if self._seq is None:
            self.__dict__["_seq"] = _SequenceCollection()
        self._seq.insert(index,item)
    def remove(self, something):
        """Remove the first occurrence of 'something' (a sequence or a module)
           Returns 'True' if the module has been removed, False if it was not found"""
        # You can remove instances of these types EDProducer, EDFilter, OutputModule,
        # EDAnalyzer, ESSource, ESProducer, Service, Sequence, SequencePlaceholder, Task,
        # _SequenceNegation, and _SequenceIgnore.
        # Mostly this is very intuitive, but there are some complications in cases
        # where objects that contain other objects are involved. See the comments
        # for the _MutatingSequenceVisitor.
        #
        # Works very similar to copyAndExclude, there are 2 differences. This changes
        # the object itself instead of making a copy and second it only removes
        # the first instance of the argument instead of all of them.
        v = _CopyAndRemoveFirstSequenceVisitor(something)
        self.visit(v)
        if v.didRemove():
            self._seq = v.result(self)[0]
            self._tasks.clear()
            self.associate(*v.result(self)[1])
        return v.didRemove()
    def resolve(self, processDict,keepIfCannotResolve=False):
        if self._seq is not None:
            self._seq = self._seq.resolve(processDict,keepIfCannotResolve)
        for task in self._tasks:
            task.resolve(processDict,keepIfCannotResolve)
        return self
    def __setattr__(self,name,value):
        if not name.startswith("_"):
            raise AttributeError("You cannot set parameters for sequence like objects.")
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
    def insertInto(self, parameterSet, myname, decoratedList):
        parameterSet.addVString(True, myname, decoratedList)
    def visit(self,visitor):
        """Passes to visitor's 'enter' and 'leave' method each item describing the module sequence.
        If the item contains 'sub' items then visitor will see those 'sub' items between the
        item's 'enter' and 'leave' calls.
        """
        if self._seq is not None:
            self._seq.visitNode(visitor)
        for item in self._tasks:
            visitor.enter(item)
            item.visit(visitor)
            visitor.leave(item)

class _UnarySequenceOperator(_BooleanLogicSequenceable):
    """For ~ and - operators"""
    def __init__(self, operand):
       self._operand = operand
       if isinstance(operand, _ModuleSequenceType):
           raise RuntimeError("This operator cannot accept a sequence")
       if not isinstance(operand, _Sequenceable):
           raise RuntimeError("This operator cannot accept a non sequenceable type")
    def __eq__(self, other):
        # allows replace(~a, b)
        return isinstance(self, type(other)) and self._operand==other._operand
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
    def resolve(self, processDict,keepIfCannotResolve=False):
        self._operand = self._operand.resolve(processDict,keepIfCannotResolve)
        return self
    def isOperation(self):
        return True
    def _visitSubNodes(self,visitor):
        self._operand.visitNode(visitor)
    def decoration(self):
        self._operand.decoration()


class _SequenceNegation(_UnarySequenceOperator):
    """Used in the expression tree for a sequence as a stand in for the '!' operator"""
    def __init__(self, operand):
        super(_SequenceNegation,self).__init__(operand)
    def __str__(self):
        return '~%s' %self._operand
    def dumpSequenceConfig(self):
        return '!%s' %self._operand.dumpSequenceConfig()
    def dumpSequencePython(self, options=PrintOptions()):
        if self._operand.isOperation():
            return '~(%s)' %self._operand.dumpSequencePython(options)
        return '~%s' %self._operand.dumpSequencePython(options)
    def decoration(self):
        return '!'

class _SequenceIgnore(_UnarySequenceOperator):
    """Used in the expression tree for a sequence as a stand in for the '-' operator"""
    def __init__(self, operand):
        super(_SequenceIgnore,self).__init__(operand)
    def __str__(self):
        return 'ignore(%s)' %self._operand
    def dumpSequenceConfig(self):
        return '-%s' %self._operand.dumpSequenceConfig()
    def dumpSequencePython(self, options=PrintOptions()):
        return 'cms.ignore(%s)' %self._operand.dumpSequencePython(options)
    def decoration(self):
        return '-'

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
    def resolve(self, processDict,keepIfCannotResolve=False):
        if not self._name in processDict:
            #print str(processDict.keys())
            if keepIfCannotResolve:
                return self
            raise RuntimeError("The SequencePlaceholder "+self._name+ " cannot be resolved.\n Known keys are:"+str(processDict.keys()))
        o = processDict[self._name]
        if not isinstance(o,_Sequenceable):
            raise RuntimeError("The SequencePlaceholder "+self._name+ " refers to an object type which is not allowed to be on a sequence: "+str(type(o)))
        return o.resolve(processDict)

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
    def dumpSequencePython(self, options=PrintOptions()):
        return 'cms.SequencePlaceholder("%s")'%self._name
    def dumpPython(self, options=PrintOptions()):
        result = 'cms.SequencePlaceholder(\"'
        if options.isCfg:
           result += 'process.'
        result += +self._name+'\")\n'
    

class Schedule(_ValidatingParameterListBase,_ConfigureComponent,_Unlabelable):

    def __init__(self,*arg,**argv):
        super(Schedule,self).__init__(*arg)
        self._tasks = OrderedSet()
        theKeys = argv.keys()
        if theKeys:
            if len(theKeys) > 1 or theKeys[0] != "tasks":
                raise RuntimeError("The Schedule constructor can only have one keyword argument after its Path and\nEndPath arguments and it must use the keyword 'tasks'")
            taskList = argv["tasks"]
            # Normally we want a list of tasks, but we let it also work if the value is one Task
            if isinstance(taskList,Task):
                self.associate(taskList)
            else:
                try:
                    # Call this just to check that taskList is a list or other iterable object
                    self.__dummy(*taskList)
                except:
                    raise RuntimeError("The Schedule constructor argument with keyword 'tasks' must have a\nlist (or other iterable object) as its value")
                if taskList:
                    self.associate(*taskList)

    def __dummy(self, *args):
        pass

    def associate(self,*tasks):
        for task in tasks:
            if not isinstance(task, Task):
                raise TypeError("The associate function in the class Schedule only works with arguments of type Task")
            self._tasks.add(task)
    @staticmethod
    def _itemIsValid(item):
        return isinstance(item,Path) or isinstance(item,EndPath)
    def copy(self):
        import copy
        aCopy = copy.copy(self)
        aCopy._tasks = OrderedSet(self._tasks)
        return aCopy
    def _place(self,label,process):
        process.setPartialSchedule_(self,label)
    def moduleNames(self):
        result = set()
        visitor = NodeNameVisitor(result)
        for seq in self:
            seq.visit(visitor)
        for t in self._tasks:
            t.visit(visitor)
        return result
    def contains(self, mod):
        visitor = ContainsModuleVisitor(mod)
        for seq in self:
            seq.visit(visitor)
            if visitor.result():
                return True
        for t in self._tasks:
            t.visit(visitor)
            if visitor.result():
                return True
        return visitor.result()
    def dumpPython(self, options=PrintOptions()):
        pathNames = ['process.'+p.label_() for p in self]
        if pathNames:
            s=', '.join(pathNames)
        else:
            s = ''
        associationContents = set()
        for task in self._tasks:
            if task.hasLabel_():
                associationContents.add(_Labelable.dumpSequencePython(task, options))
            else:
                associationContents.add(task.dumpPythonNoNewline(options))
        taskStrings = list()
        for iString in sorted(associationContents):
            taskStrings.append(iString)
        if taskStrings and s:
            return 'cms.Schedule(*[ ' + s + ' ], tasks=[' + ', '.join(taskStrings) + '])\n'
        elif s:
            return 'cms.Schedule(*[ ' + s + ' ])\n'
        elif taskStrings:
            return 'cms.Schedule(tasks=[' + ', '.join(taskStrings) + '])\n'
        else:
            return 'cms.Schedule()\n'

    def __str__(self):
        return self.dumpPython()

# Fills a list of all Sequences visited
# Can visit a Sequence, Path, or EndPath
class SequenceVisitor(object):
    def __init__(self,d):
        self.deps = d
    def enter(self,visitee):
        if isinstance(visitee,Sequence):
            self.deps.append(visitee)
        pass
    def leave(self,visitee):
        pass

# Fills a list of all Tasks visited
# Can visit a Task, Sequence, Path, or EndPath
class TaskVisitor(object):
    def __init__(self,d):
        self.deps = d
    def enter(self,visitee):
        if isinstance(visitee,Task):
            self.deps.append(visitee)
        pass
    def leave(self,visitee):
        pass

# Fills a list of all modules visited.
# Can visit a Sequence, Path, EndPath, or Task
# For purposes of this visitor, a module is considered
# to be an object that is one of these types: EDProducer,
# EDFilter, EDAnalyzer, OutputModule, ESProducer, ESSource,
# Service. The last three of these can only appear on a
# Task, they are not sequenceable. An object of one
# of these types is also called a leaf.
class ModuleNodeVisitor(object):
    def __init__(self,l):
        self.l = l
    def enter(self,visitee):
        if visitee.isLeaf():
            self.l.append(visitee)
        pass
    def leave(self,visitee):
        pass

# Should not be used on Tasks.
# Similar to ModuleNodeVisitor with the following
# differences. It only lists the modules that were
# contained inside a Task.  It should only be used
# on Sequences, Paths, and EndPaths.
class ModuleNodeOnTaskVisitor(object):
    def __init__(self,l):
        self.l = l
        self._levelInTasks = 0
    def enter(self,visitee):
        if isinstance(visitee, Task):
            self._levelInTasks += 1
        if self._levelInTasks == 0:
            return
        if visitee.isLeaf():
            self.l.append(visitee)
        pass
    def leave(self,visitee):
        if self._levelInTasks > 0:
            if isinstance(visitee, Task):
                self._levelInTasks -= 1

# Should not be used on Tasks.
# Similar to ModuleNodeVisitor with the following
# differences. It only lists the modules that were
# outside a Task, in the sequenced part of the sequence.
# It should only be used on Sequences, Paths, and
# EndPaths.
class ModuleNodeNotOnTaskVisitor(object):
    def __init__(self,l):
        self.l = l
        self._levelInTasks = 0
    def enter(self,visitee):
        if isinstance(visitee, Task):
            self._levelInTasks += 1
        if self._levelInTasks > 0:
            return
        if visitee.isLeaf():
            self.l.append(visitee)
        pass
    def leave(self,visitee):
        if self._levelInTasks > 0:
            if isinstance(visitee, Task):
                self._levelInTasks -= 1

# Can visit Tasks, Sequences, Paths, and EndPaths
# result will be set to True if and only if
# the module is in the object directly or
# indirectly through contained Sequences or
# associated Tasks.
class ContainsModuleVisitor(object):
    def __init__(self,mod):
        self._mod = mod
        self._result = False

    def result(self):
        return self._result

    def enter(self,visitee):
        if self._mod is visitee:
            self._result = True

    def leave(self,visitee):
        pass

# Can visit Tasks, Sequences, Paths, and EndPaths
# Fills a set of the names of the visited leaves.
# For the labelable ones the name is the label.
# For a Service the name is the type.
# It raises an exception if a labelable object
# does not have a label at all. It will return
# 'None' if the label attribute exists but was set
# to None. If a Service is not attached to the process
# it will also raise an exception.
class NodeNameVisitor(object):
    """ takes a set as input"""
    def __init__(self,l):
        self.l = l
    def enter(self,visitee):
        if visitee.isLeaf():
            if isinstance(visitee, _Labelable):
                self.l.add(visitee.label_())
            else:
                if visitee._inProcess:
                    self.l.add(visitee.type_())
                else:
                    raise RuntimeError("Service not attached to process")
    def leave(self,visitee):
        pass

# This visitor works only with Sequences, Paths and EndPaths
# It will not work on Tasks
class ExpandVisitor(object):
    """ Expands the sequence into leafs and UnaryOperators """
    def __init__(self, type):
        self._type = type
        self.l = []
        self.taskLeaves = []
        self._levelInTasks = 0
    def enter(self,visitee):
        if isinstance(visitee, Task):
            self._levelInTasks += 1
            return
        if visitee.isLeaf():
            if self._levelInTasks > 0:
                self.taskLeaves.append(visitee)
            else:
                self.l.append(visitee)
    def leave(self, visitee):
        if self._levelInTasks > 0:
            if isinstance(visitee, Task):
                self._levelInTasks -= 1
            return
        if isinstance(visitee,_UnarySequenceOperator):
            self.l[-1] = visitee
    def result(self):
        # why doesn't (sum(self.l) work?
        seq = self.l[0]
        if len(self.l) > 1:
            for el in self.l[1:]:
                seq += el
        return self._type(seq, Task(*self.taskLeaves))
    def resultString(self):
        sep = ''
        returnValue = ''
        for m in self.l:
            if m is not None:
                returnValue += sep+str(m)
                sep = '+'
        if returnValue:
            sep = ','
        for n in self.taskLeaves:
            if n is not None:
                returnValue += sep+str(n)
            sep = ','
        return returnValue

    
# This visitor is only meant to run on Sequences, Paths, and EndPaths
# It intentionally ignores nodes on Tasks when it does this.
class DecoratedNodeNameVisitor(object):
    """ Adds any '!' or '-' needed.  Takes a list """
    def __init__(self,l):
        self.l = l
        self._decoration =''
        self._levelInTasks = 0

    def initialize(self):
        self.l[:] = []
        self._decoration =''
        self._levelInTasks = 0

    def enter(self,visitee):
        if isinstance(visitee, Task):
            self._levelInTasks += 1
        if self._levelInTasks > 0:
            return
        if visitee.isLeaf():
            if hasattr(visitee, "_Labelable__label"):
                self.l.append(self._decoration+visitee.label_())
            else:
                error = "An object in a sequence was not found in the process\n"
                if hasattr(visitee, "_filename"):
                    error += "From file " + visitee._filename
                else:
                    error += "Dump follows\n" + repr(visitee)
                raise RuntimeError(error)
        if isinstance(visitee,_BooleanLogicExpression):
            self.l.append(self._decoration+visitee.operatorString())
        if isinstance(visitee,_UnarySequenceOperator):
            self._decoration=visitee.decoration()
        else:
            self._decoration=''

    def leave(self,visitee):
        # Ignore if this visitee is inside a Task
        if self._levelInTasks > 0:
            if isinstance(visitee, Task):
                self._levelInTasks -= 1
            return
        if isinstance(visitee,_BooleanLogicExpression):
            #need to add the 'go back' command to keep track of where we are in the tree
            self.l.append('@')

# This visitor is only meant to run on Sequences, Paths, and EndPaths
# Similar to DecoratedNodeNameVistor. The only difference
# is it also builds a separate list of leaves on Tasks.
class DecoratedNodeNamePlusVisitor(object):
    """ Adds any '!' or '-' needed.  Takes a list """
    def __init__(self,l):
        self.l = l
        self._decoration =''
        self._levelInTasks = 0
        self._leavesOnTasks = []

    def initialize(self):
        self.l[:] = []
        self._decoration =''
        self._levelInTasks = 0
        self._leavesOnTasks[:] = []

    def enter(self,visitee):
        if isinstance(visitee, Task):
            self._levelInTasks += 1
        if self._levelInTasks > 0:
            if visitee.isLeaf():
                self._leavesOnTasks.append(visitee)
            return
        if visitee.isLeaf():
            if hasattr(visitee, "_Labelable__label"):
                self.l.append(self._decoration+visitee.label_())
            else:
                error = "An object in a sequence was not found in the process\n"
                if hasattr(visitee, "_filename"):
                    error += "From file " + visitee._filename
                else:
                    error += "Dump follows\n" + repr(visitee)
                raise RuntimeError(error)
        if isinstance(visitee,_BooleanLogicExpression):
            self.l.append(self._decoration+visitee.operatorString())
        if isinstance(visitee,_UnarySequenceOperator):
            self._decoration=visitee.decoration()
        else:
            self._decoration=''

    def leave(self,visitee):
        # Ignore if this visitee is inside a Task
        if self._levelInTasks > 0:
            if isinstance(visitee, Task):
                self._levelInTasks -= 1
            return
        if isinstance(visitee,_BooleanLogicExpression):
            #need to add the 'go back' command to keep track of where we are in the tree
            self.l.append('@')

    def leavesOnTasks(self):
        return self._leavesOnTasks

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
       if visitee.isLeaf():
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
       if not visitee.isLeaf():
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
       if not visitee.isLeaf():
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


# This visitor can also be used on Tasks.
class _MutatingSequenceVisitor(object):
    """Traverses a Sequence and constructs a new sequence by applying the operator to each element of the sequence"""

    # In many cases this operates in an intuitive manner that needs
    # no explanation, but there are some complex cases and I will try to
    # explain these in the following comments.
    #
    # First of all the top level Sequence or Task being visited may contain
    # many objects of different types. These contained objects are never
    # modified. If they are not left the same, they are instead replaced
    # by other instances, replaced by new instances or removed.
    # Contained objects are only replaced or removed when they were directly
    # modified or if they contain something that was modified.
    # If all the contents of a Sequence, Task, _SequenceNegation or _SequenceIgnore
    # object that is not at the top level are removed, then the containing
    # object is also removed.
    # If the contents of a Sequence other than the top level sequence are
    # modified, then the sequence elements and Task objects it contains get
    # passed up to be included in the top level sequence. If the contents of
    # a Task are modified, a new Task object is created and passed up to be
    # included in the top level Sequence or Task. If it is a _SequenceNegation
    # or _SequenceIgnore instance it will simply be removed completely if its
    # operand is removed. If the operand is replaced then a new object of the
    # same type will be constructed replacing the old.
    #
    # Note that if a Sequence contains a SequencePlaceholder, the future contents
    # of that placeholder are not affected by the changes. If that is an issue,
    # then you probably want to resolve the placeholders before using this
    # class.
    #
    # If this is used multiple times on the same sequence or task, the consequences
    # might interfere with one another in unusual cases.
    #
    # One example, the matching to find objects to modify is based on instances
    # (the python id) being the same. So if you modify the contents of a Task or
    # Sequence and then subsequently try to modify that Sequence or Task, then
    # it will either no longer exist or be a different instance and so nothing
    # would get modified.  Note that the one exception to this matching by instance
    # is _SequenceIgnore and _SequenceNegation. In that case, two objects are
    # recognized as matching if the contained module is the same instance instead
    # of requiring the _SequenceNegation or _SequenceIgnore object to be the same
    # instance.
    #
    # Another example. There is an input operator that removes the first instance
    # of an object. Applying this visitor with that operation might give unexpected
    # results if another operation previously changed the number of times the
    # that instance appears or the order it appears in the visitation. This
    # should only be an issue if the item is on a Task and even then only in
    # unusual circumstances.

    def __init__(self,operator):
        self.__operator = operator
        # You add a list to the __stack when entering any non-Leaf object
        # and pop the last element when leaving any non-Leaf object
        self.__stack = list()
        self.__stack.append(list())
        self.__didApply = False
        self.__levelInModifiedNonLeaf = 0
    def enter(self,visitee):
        # Ignore the content of replaced or removed Sequences,
        # Tasks, and operators.
        if self.__levelInModifiedNonLeaf > 0:
            if not visitee.isLeaf():
                self.__levelInModifiedNonLeaf += 1
            return

        # Just a sanity check
        if not len(self.__stack) > 0:
            raise RuntimeError("LogicError Empty stack in MutatingSequenceVisitor.\n"
                               "This should never happen. Contact a Framework developer.")

        # The most important part.
        # Apply the operator that might change things, The rest
        # of the class is just dealing with side effects of these changes.
        v = self.__operator(visitee)

        if v is visitee:
            # the operator did not change the visitee
            # The 3 element list being appended has the following contents
            # element 0 - either the unmodified object, the modified object, or
            #   a sequence collection when it is a Sequence whose contents have
            #   been modified.
            # element 1 - Indicates whether the object was modified.
            # element 2 - None or a list of tasks for a Sequence
            #   whose contents have been modified.
            self.__stack[-1].append([visitee, False, None])
            if not visitee.isLeaf():
                # need to add a list to keep track of the contents
                # of the Sequence, Task, or operator we just entered.
                self.__stack.append(list())
        else:
            # the operator changed the visitee
            self.__didApply = True
            self.__stack[-1].append([v, True, None])
            if not visitee.isLeaf():
                # Set flag to indicate modified Sequence, Task, or operator
                self.__levelInModifiedNonLeaf = 1
    def leave(self,visitee):

        # nothing to do for leaf types because they do not have contents
        if visitee.isLeaf():
            return

        # Ignore if this visitee is inside something that was already removed
        # or replaced.
        if self.__levelInModifiedNonLeaf > 0:
            self.__levelInModifiedNonLeaf -= 1
            return

        # Deal with visitees which have contents (Sequence, Task, _SequenceIgnore,
        # or _SequenceNegation) and although we know the visitee itself did not get
        # changed by the operator, the contents of the visitee might have been changed.

        # did any object inside the visitee change?
        contents = self.__stack[-1]
        changed = False
        allNull = True
        for c in contents:
            if c[1] == True:
                changed = True
            if c[0] is not None:
                allNull = False
        if changed:
            if allNull:
                self.__stack[-2][-1] = [None, True, None]

            elif isinstance(visitee, _UnarySequenceOperator):
                node = visitee.__new__(type(visitee))
                node.__init__(contents[0][0])
                self.__stack[-2][-1] = [node, True, None]

            elif isinstance(visitee, Task):
                nonNull = []
                for c in contents:
                    if c[0] is not None:
                        nonNull.append(c[0])
                self.__stack[-2][-1] = [Task(*nonNull), True, None]
            elif isinstance(visitee, Sequence):
                seq = _SequenceCollection()
                tasks = list()
                for c in contents:
                    if c[0] is None:
                        continue
                    if isinstance(c[0], Task):
                        tasks.append(c[0])
                    else:
                        seq = seq + c[0]
                        if c[2] is not None:
                            tasks.extend(c[2])

                self.__stack[-2][-1] = [seq, True, tasks]

        # When you exit the Sequence, Task, or operator,
        # drop the list which holds information about
        # its contents.
        if not visitee.isLeaf():
            self.__stack = self.__stack[:-1]

    def result(self, visitedContainer):

        if isinstance(visitedContainer, Task):
            result = list()
            for n in (x[0] for x in self.__stack[0]):
                if n is not None:
                    result.append(n)
            return result

        seq = _SequenceCollection()
        tasks = list()
        for c in self.__stack[0]:
            if c[0] is None:
                continue
            if isinstance(c[0], Task):
                tasks.append(c[0])
            else:
                seq = seq + c[0]
                if c[2] is not None:
                    tasks.extend(c[2])
        return [seq, tasks]

    def _didApply(self):
        return self.__didApply

# This visitor can also be used on Tasks.
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

# This visitor can also be used on Tasks.
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

# This visitor can also be used on Tasks.
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

class Task(_ConfigureComponent, _Labelable) :
    """Holds EDProducers, EDFilters, ESProducers, ESSources, Services, and Tasks.
    A Task can be associated with Sequences, Paths, EndPaths and the Schedule.
    An EDProducer or EDFilter will be enabled to run unscheduled if it is on
    a task associated with the Schedule or any scheduled Path or EndPath (directly
    or indirectly through Sequences) and not be on any scheduled Path or EndPath.
    ESSources, ESProducers, and Services will be enabled to run if they are on
    a Task associated with the Schedule or a scheduled Path or EndPath.  In other
    cases, they will be enabled to run if and only if they are not on a Task attached
    to the process.
    """

    def __init__(self, *items):
        self._collection = OrderedSet()
        self.add(*items)

    def __setattr__(self,name,value):
        if not name.startswith("_"):
            raise AttributeError("You cannot set parameters for Task objects.")
        else:
            self.__dict__[name] = value

    def add(self, *items):
        for item in items:
            if not isinstance(item, _ConfigureComponent) or not item._isTaskComponent():
                if not isinstance(item, TaskPlaceholder):
                    raise RuntimeError("Adding an entry of type '" + type(item).__name__ + "'to a Task.\n"
                                       "It is illegal to add this type to a Task.")
            self._collection.add(item)

    def _place(self, name, proc):
        proc._placeTask(name,self)

    def fillContents(self, taskContents, options=PrintOptions()):
        # only dump the label, if possible
        if self.hasLabel_():
            taskContents.add(_Labelable.dumpSequencePython(self, options))
        else:
            for i in self._collection:
                if isinstance(i, Task):
                    i.fillContents(taskContents, options)
                else:
                    taskContents.add(i.dumpSequencePython(options))

    def dumpPython(self, options=PrintOptions()):
        s = self.dumpPythonNoNewline(options)
        return s + "\n"

    def dumpPythonNoNewline(self, options=PrintOptions()):
        """Returns a string which is the python representation of the object"""
        taskContents = set()
        for i in self._collection:
            if isinstance(i, Task):
                i.fillContents(taskContents, options)
            else:
                taskContents.add(i.dumpSequencePython(options))
        s=''
        iFirst = True
        for item in sorted(taskContents):
            if not iFirst:
                s += ", "
            iFirst = False
            s += item
        if len(taskContents) > 255:
            return "cms.Task(*[" + s + "])"
        return "cms.Task(" + s + ")"

    def _isTaskComponent(self):
        return True

    def isLeaf(self):
        return False

    def visit(self,visitor):
        for i in self._collection:
            visitor.enter(i)
            if not i.isLeaf():
                i.visit(visitor)
            visitor.leave(i)

    def _errorstr(self):
        return "Task(...)"

    def __iter__(self):
        for key in self._collection:
            yield key

    def __str__(self):
        l = []
        v = ModuleNodeVisitor(l)
        self.visit(v)
        s = ''
        for i in l:
            if s:
                s += ', '
            s += str (i)
        return s

    def __repr__(self):
        s = str(self)
        return "cms."+type(self).__name__+'('+s+')\n'

    def moduleNames(self):
        """Returns a set containing the names of all modules being used"""
        result = set()
        visitor = NodeNameVisitor(result)
        self.visit(visitor)
        return result
    def contains(self, mod):
        visitor = ContainsModuleVisitor(mod)
        self.visit(visitor)
        return visitor.result()
    def copy(self):
        return Task(*self._collection)
    def copyAndExclude(self,listOfModulesToExclude):
        """Returns a copy of the sequence which excludes those module in 'listOfModulesToExclude'"""
        # You can exclude instances of these types EDProducer, EDFilter, ESSource, ESProducer,
        # Service, or Task.
        # Mostly this is very intuitive, but there are some complications in cases
        # where objects that contain other objects are involved. See the comments
        # for the _MutatingSequenceVisitor.
        for i in listOfModulesToExclude:
            if not i._isTaskComponent():
                raise TypeError("copyAndExclude can only exclude objects that can be placed on a Task")
        v = _CopyAndExcludeSequenceVisitor(listOfModulesToExclude)
        self.visit(v)
        return Task(*v.result(self))
    def expandAndClone(self):
        # Name of this function is not very good. It makes a shallow copy with all
        # the subTasks flattened out (removed), but keeping all the
        # modules that were in those subTasks as well as the top level
        # ones.
        l = []
        v = ModuleNodeVisitor(l)
        self.visit(v)
        return Task(*l)
    def replace(self, original, replacement):
        """Finds all instances of 'original' and substitutes 'replacement' for them.
           Returns 'True' if a replacement occurs."""
        # This works for either argument being of type EDProducer, EDFilter, ESProducer,
        # ESSource, Service, or Task.
        #
        # Mostly this is very intuitive, but there are some complications in cases
        # where objects that contain other objects are involved. See the comments
        # for the _MutatingSequenceVisitor.

        if not original._isTaskComponent() or (not replacement is None and not replacement._isTaskComponent()):
           raise TypeError("The Task replace function only works with objects that can be placed on a Task\n" + \
                           "           replace was called with original type = " + str(type(original)) + "\n" + \
                           "           and replacement type = " + str(type(replacement)) + "\n")
        else:
            v = _CopyAndReplaceSequenceVisitor(original,replacement)
            self.visit(v)
            if v.didReplace():
                self._collection.clear()
                self.add(*v.result(self))
            return v.didReplace()

    def remove(self, something):
        """Remove the first occurrence of a module
           Returns 'True' if the module has been removed, False if it was not found"""
        # You can remove instances of these types EDProducer, EDFilter, ESSource,
        # ESProducer, Service, or Task,
        #
        # Mostly this is very intuitive, but there are some complications in cases
        # where objects that contain other objects are involved. See the comments
        # for the _MutatingSequenceVisitor.
        #
        # Works very similar to copyAndExclude, there are 2 differences. This changes
        # the object itself instead of making a copy and second it only removes
        # the first instance of the argument instead of all of them.
        if not something._isTaskComponent():
           raise TypeError("remove only works with objects that can be placed on a Task")
        v = _CopyAndRemoveFirstSequenceVisitor(something)
        self.visit(v)
        if v.didRemove():
            self._collection.clear()
            self.add(*v.result(self))
        return v.didRemove()

    def resolve(self, processDict,keepIfCannotResolve=False):
        temp = OrderedSet()
        for i in self._collection:
            if isinstance(i, Task) or isinstance(i, TaskPlaceholder):
                temp.add(i.resolve(processDict,keepIfCannotResolve))
            else:
                temp.add(i)
        self._collection = temp
        return self

class TaskPlaceholder(object):
    def __init__(self, name):
        self._name = name
    def _isTaskComponent(self):
        return True
    def isLeaf(self):
        return False
    def visit(self,visitor):
        pass
    def __str__(self):
        return self._name
    def insertInto(self, parameterSet, myname):
        raise RuntimeError("The TaskPlaceholder "+self._name
                           +" was never overridden")
    def resolve(self, processDict,keepIfCannotResolve=False):
        if not self._name in processDict:
            if keepIfCannotResolve:
                return self
            raise RuntimeError("The TaskPlaceholder "+self._name+ " cannot be resolved.\n Known keys are:"+str(processDict.keys()))
        o = processDict[self._name]
        if not o._isTaskComponent():
            raise RuntimeError("The TaskPlaceholder "+self._name+ " refers to an object type which is not allowed to be on a task: "+str(type(o)))
        if isinstance(o, Task):
            return o.resolve(processDict)
        return o
    def copy(self):
        returnValue =TaskPlaceholder.__new__(type(self))
        returnValue.__init__(self._name)
        return returnValue
    def dumpSequencePython(self, options=PrintOptions()):
        return 'cms.TaskPlaceholder("%s")'%self._name
    def dumpPython(self, options=PrintOptions()):
        result = 'cms.TaskPlaceholder(\"'
        if options.isCfg:
           result += 'process.'
        result += +self._name+'\")\n'

if __name__=="__main__":
    import unittest
    class DummyModule(_Labelable, _SequenceLeaf, _ConfigureComponent):
        def __init__(self,name):
            self.setLabel(name)
        def _isTaskComponent(self):
            return True
        def __repr__(self):
            return self.label_()
    class DummyBooleanModule(_Labelable, _BooleanLogicSequenceLeaf):
        def __init__(self,name):
            self.setLabel(name)
    class TestModuleCommand(unittest.TestCase):
        def setUp(self):
            """Nothing to do """
            pass
        def testBoolean(self):
            a = DummyBooleanModule("a")
            b = DummyBooleanModule("b")
            p = Path( a & b)
            self.assertEqual(p.dumpPython(None),"cms.Path(process.a&process.b)\n")
            l = list()
            namesVisitor = DecoratedNodeNameVisitor(l)
            p.visit(namesVisitor)
            self.assertEqual(l,['&','a','b','@'])
            p2 = Path( a | b)
            self.assertEqual(p2.dumpPython(None),"cms.Path(process.a|process.b)\n")
            l[:]=[]
            p2.visit(namesVisitor)
            self.assertEqual(l,['|','a','b','@'])
            c = DummyBooleanModule("c")
            d = DummyBooleanModule("d")
            p3 = Path(a & b & c & d)
            self.assertEqual(p3.dumpPython(None),"cms.Path(process.a&process.b&process.c&process.d)\n")
            l[:]=[]
            p3.visit(namesVisitor)
            self.assertEqual(l,['&','a','b','c','d','@'])
            p3 = Path(((a & b) & c) & d)
            self.assertEqual(p3.dumpPython(None),"cms.Path(process.a&process.b&process.c&process.d)\n")
            p3 = Path(a & (b & (c & d)))
            self.assertEqual(p3.dumpPython(None),"cms.Path(process.a&process.b&process.c&process.d)\n")
            p3 = Path((a & b) & (c & d))
            self.assertEqual(p3.dumpPython(None),"cms.Path(process.a&process.b&process.c&process.d)\n")
            p3 = Path(a & (b & c) & d)
            self.assertEqual(p3.dumpPython(None),"cms.Path(process.a&process.b&process.c&process.d)\n")
            p4 = Path(a | b | c | d)
            self.assertEqual(p4.dumpPython(None),"cms.Path(process.a|process.b|process.c|process.d)\n")
            p5 = Path(a | b & c & d )
            self.assertEqual(p5.dumpPython(None),"cms.Path(process.a|(process.b&process.c&process.d))\n")
            l[:]=[]
            p5.visit(namesVisitor)
            self.assertEqual(l,['|','a','&','b','c','d','@','@'])
            p5 = Path(a & b | c & d )
            self.assertEqual(p5.dumpPython(None),"cms.Path((process.a&process.b)|(process.c&process.d))\n")
            l[:]=[]
            p5.visit(namesVisitor)
            self.assertEqual(l,['|','&','a','b','@','&','c','d','@','@'])
            p5 = Path(a & (b | c) & d )
            self.assertEqual(p5.dumpPython(None),"cms.Path(process.a&(process.b|process.c)&process.d)\n")
            l[:]=[]
            p5.visit(namesVisitor)
            self.assertEqual(l,['&','a','|','b','c','@','d','@'])
            p5 = Path(a & b & c | d )
            self.assertEqual(p5.dumpPython(None),"cms.Path((process.a&process.b&process.c)|process.d)\n")
            l[:]=[]
            p5.visit(namesVisitor)
            self.assertEqual(l,['|','&','a','b','c','@','d','@'])
            p6 = Path( a & ~b)
            self.assertEqual(p6.dumpPython(None),"cms.Path(process.a&(~process.b))\n")
            l[:]=[]
            p6.visit(namesVisitor)
            self.assertEqual(l,['&','a','!b','@'])
            p6 = Path( a & ignore(b))
            self.assertEqual(p6.dumpPython(None),"cms.Path(process.a&(cms.ignore(process.b)))\n")
            l[:]=[]
            p6.visit(namesVisitor)
            self.assertEqual(l,['&','a','-b','@'])
            p6 = Path(~(a&b))
            self.assertEqual(p6.dumpPython(None),"cms.Path(~(process.a&process.b))\n")
            l[:]=[]
            p6.visit(namesVisitor)
            self.assertEqual(l,['!&','a','b','@'])

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
            t1 = Task(a)
            t2 = Task(c, b)
            t3 = Task()
            p9 = Path((a+b)*c, t1)
            self.assertEqual(p9.dumpPython(None),"cms.Path(process.a+process.b+process.c, cms.Task(process.a))\n")
            p10 = Path((a+b)*c, t2, t1)
            self.assertEqual(p10.dumpPython(None),"cms.Path(process.a+process.b+process.c, cms.Task(process.a), cms.Task(process.b, process.c))\n")
            p11 = Path(t1, t2, t3)
            self.assertEqual(p11.dumpPython(None),"cms.Path(cms.Task(), cms.Task(process.a), cms.Task(process.b, process.c))\n")
            d = DummyModule("d")
            e = DummyModule('e')
            f = DummyModule('f')
            t4 = Task(d, Task(f))
            s = Sequence(e, t4)
            p12 = Path(a+b+s+c,t1)
            self.assertEqual(p12.dumpPython(None),"cms.Path(process.a+process.b+cms.Sequence(process.e, cms.Task(process.d, process.f))+process.c, cms.Task(process.a))\n")
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
            p10.visit(namesVisitor)
            self.assertEqual(l, ['a', 'b', 'c'])
            l[:] = []
            p12.visit(namesVisitor)
            self.assertEqual(l, ['a', 'b', 'e', 'c'])
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

            c=DummyModule("c")
            d=DummyModule("d")
            e=DummyModule("e")
            f=DummyModule("f")
            g=DummyModule("g")
            t1 = Task(d)
            t2 = Task(e, t1)
            t3 = Task(f, g, t2)
            s=Sequence(plusAB, t3, t2)
            multSC = s*c
            p=Path(multSC, t1, t2)

            l = []
            v = ModuleNodeVisitor(l)
            p.visit(v)
            expected = [a,b,f,g,e,d,e,d,c,d,e,d]

            l[:] = []
            v = ModuleNodeOnTaskVisitor(l)
            p.visit(v)
            expected = [f,g,e,d,e,d,d,e,d]
            self.assertEqual(expected,l)

            l[:] = []
            v = ModuleNodeNotOnTaskVisitor(l)
            p.visit(v)
            expected = [a,b,c]
            self.assertEqual(expected,l)


            t=TestVisitor(enters=[s,a,b,t3,f,g,t2,e,t1,d,t2,e,t1,d,c,t1,d,t2,e,t1,d],
                          leaves=[a,b,f,g,e,d,t1,t2,t3,e,d,t1,t2,s,c,d,t1,e,d,t1,t2])
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
            l[:]=[]
            s1 = Sequence(m1)
            s2 = SequencePlaceholder("s3")
            s3 = Sequence(m2)
            s4 = SequencePlaceholder("s2")
            p=Path(s1+s4)
            p.resolve(d)
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
            s3.replace(m2, ~m1)
            l[:] = []
            s3.visit(namesVisitor)
            self.assertEqual(l, ['!m1', 'm1', '!m1'])

            s3 = Sequence(ignore(m1)*s2)
            s3.replace(ignore(m1), m2)
            l[:] = []
            s3.visit(namesVisitor)
            self.assertEqual(l, ['m2', 'm1', 'm2'])
            s3.replace(m2, ignore(m1))
            l[:] = []
            s3.visit(namesVisitor)
            self.assertEqual(l, ['-m1', 'm1', '-m1'])

            ph = SequencePlaceholder('x')
            s4 = Sequence(Sequence(ph))
            s4.replace(ph,m2)
            self.assertEqual(s4.dumpPython(None), "cms.Sequence(process.m2)\n")

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

            m6 = DummyModule("m6")
            m7 = DummyModule("m7")
            m8 = DummyModule("m8")
            m9 = DummyModule("m9")

            t6 = Task(m6)
            t7 = Task(m7)
            t89 = Task(m8, m9)

            s1 = Sequence(m1+m2, t6)
            s2 = Sequence(m3+m4, t7)
            s3 = Sequence(s1+s2, t89)
            s3.replace(m3,m5)
            l[:] = []
            s3.visit(namesVisitor)
            self.assertEqual(l,['m1','m2','m5','m4'])

            s3.replace(m8,m1)
            self.assertTrue(s3.dumpPython(None) == "cms.Sequence(cms.Sequence(process.m1+process.m2, cms.Task(process.m6))+process.m5+process.m4, cms.Task(process.m1, process.m9), cms.Task(process.m7))\n")

            s3.replace(m1,m7)
            self.assertTrue(s3.dumpPython(None) == "cms.Sequence(process.m7+process.m2+process.m5+process.m4, cms.Task(process.m6), cms.Task(process.m7), cms.Task(process.m7, process.m9))\n")
            result = s3.replace(t7, t89)
            self.assertTrue(s3.dumpPython(None) == "cms.Sequence(process.m7+process.m2+process.m5+process.m4, cms.Task(process.m6), cms.Task(process.m7, process.m9), cms.Task(process.m8, process.m9))\n")
            self.assertTrue(result)
            result = s3.replace(t7, t89)
            self.assertFalse(result)

            t1 = Task()
            t1.replace(m1,m2)
            self.assertTrue(t1.dumpPython(None) == "cms.Task()\n")

            t1 = Task(m1)
            t1.replace(m1,m2)
            self.assertTrue(t1.dumpPython(None) == "cms.Task(process.m2)\n")

            t1 = Task(m1,m2, m2)
            t1.replace(m2,m3)
            self.assertTrue(t1.dumpPython(None) == "cms.Task(process.m1, process.m3)\n")

            t1 = Task(m1,m2)
            t2 = Task(m1,m3,t1)
            t2.replace(m1,m4)
            self.assertTrue(t2.dumpPython(None) == "cms.Task(process.m2, process.m3, process.m4)\n")

            t1 = Task(m2)
            t2 = Task(m1,m3,t1)
            t2.replace(m1,m4)
            self.assertTrue(t2.dumpPython(None) == "cms.Task(process.m2, process.m3, process.m4)\n")

            t1 = Task(m2)
            t2 = Task(m1,m3,t1)
            t2.replace(t1,m4)
            self.assertTrue(t2.dumpPython(None) == "cms.Task(process.m1, process.m3, process.m4)\n")

            t1 = Task(m2)
            t2 = Task(m1,m3,t1)
            t3 = Task(m5)
            t2.replace(m2,t3)
            self.assertTrue(t2.dumpPython(None) == "cms.Task(process.m1, process.m3, process.m5)\n")

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

            s = Sequence()
            s.insert(0, m1)
            self.assertEqual(s.index(m1),0)

            p = Path()
            p.insert(0, m1)
            self.assertEqual(s.index(m1),0)
        
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

            m6 = DummyModule("m6")
            m7 = DummyModule("m7")
            m8 = DummyModule("m8")
            m9 = DummyModule("m9")
            p = Path(s1+s3, Task(m6))
            p2 = p.expandAndClone()
            l[:] = []
            p2.visit(namesVisitor)
            self.assertEqual(l, ['m1', '!m2', 'm1', 'm2', '-m2', '!m1', 'm1', 'm2'])
            self.assertTrue(p2.dumpPython(None) == "cms.Path(process.m1+~process.m2+process.m1+process.m2+cms.ignore(process.m2)+~process.m1+process.m1+process.m2, cms.Task(process.m6))\n")

            s2 = Sequence(m1*m2, Task(m9))
            s3 = Sequence(~m1*s2)
            t8 = Task(m8)
            t8.setLabel("t8")
            p = Path(s1+s3, Task(m6, Task(m7, t8)))
            p2 = p.expandAndClone()
            l[:] = []
            p2.visit(namesVisitor)
            self.assertEqual(l, ['m1', '!m2', 'm1', 'm2', '-m2', '!m1', 'm1', 'm2'])
            self.assertTrue(p2.dumpPython(None) == "cms.Path(process.m1+~process.m2+process.m1+process.m2+cms.ignore(process.m2)+~process.m1+process.m1+process.m2, cms.Task(process.m6, process.m7, process.m8, process.m9))\n")

            t1 = Task(m1,m2)
            t2 = Task(m1,m3,t1)
            t3 = t2.expandAndClone()
            self.assertTrue(t3.dumpPython(None) == "cms.Task(process.m1, process.m2, process.m3)\n")
            t4 = Task()
            t5 = t4.expandAndClone()
            self.assertTrue(t5.dumpPython(None) == "cms.Task()\n")
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
            m4 = DummyModule("m4")
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
            s1 = Sequence(m1+m2, Task(m3), Task(m4))
            s1.remove(m4)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m2, cms.Task(process.m3))\n")
            s1 = Sequence(m1+m2+Sequence(Task(m3,m4), Task(m3), Task(m4)))
            s1.remove(m4)
            self.assertEqual(s1.dumpPython(None), "cms.Sequence(process.m1+process.m2, cms.Task(process.m3), cms.Task(process.m4))\n")
            t1 = Task(m1)
            t1.setLabel("t1")
            t2 = Task(m2,t1)
            t2.setLabel("t2")
            t3 = Task(t1,t2,m1)
            t3.remove(m1)
            self.assertTrue(t3.dumpPython(None) == "cms.Task(process.m1, process.t2)\n")
            t3.remove(m1)
            self.assertTrue(t3.dumpPython(None) == "cms.Task(process.m1, process.m2)\n")
            t3.remove(m1)
            self.assertTrue(t3.dumpPython(None) == "cms.Task(process.m2)\n")
            t3.remove(m2)
            self.assertTrue(t3.dumpPython(None) == "cms.Task()\n")

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
            s3 = s.copyAndExclude([c])
            s2.remove(a)
            self.assertEqual(s3.dumpPython(None),"cms.Sequence((process.b)+process.d)\n")
            s4 = s.copyAndExclude([a,b])
            seqs = []
            sequenceVisitor = SequenceVisitor(seqs)
            s.visit(sequenceVisitor)
            self.assertEqual(len(seqs),1)
            seqs[:] = []
            s4.visit(sequenceVisitor)
            self.assertEqual(len(seqs),0)
            self.assertEqual(s4.dumpPython(None),"cms.Sequence(process.c+process.d)\n")
            holder = SequencePlaceholder("x")
            s3 = Sequence(b+d,Task(a))
            s2 = Sequence(a+b+holder+s3)
            s = Sequence(c+s2+d)
            seqs[:] = []
            s.visit(sequenceVisitor)
            self.assertTrue(seqs == [s2,s3])
            s2 = Sequence(a+b+holder)
            s = Sequence(c+s2+d)
            self.assertEqual(s.copyAndExclude([holder]).dumpPython(None),"cms.Sequence(process.c+process.a+process.b+process.d)\n")
            s2 = Sequence(a+b+c)
            s = Sequence(s2+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence((process.a+process.b+process.c))\n")
            self.assertEqual(s.copyAndExclude([s2]).dumpPython(None),"cms.Sequence(process.d)\n")
            s2 = Sequence(a+b+c)
            s = Sequence(s2*d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence((process.a+process.b+process.c))\n")
            self.assertEqual(s.copyAndExclude([a,b,c]).dumpPython(None),"cms.Sequence(process.d)\n")
            s = Sequence(ignore(a)+b+c+d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([ignore(a)]).dumpPython(None),"cms.Sequence(process.b+process.c+process.d)\n")
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
            self.assertEqual(s.copyAndExclude([~b]).dumpPython(None),"cms.Sequence(process.a+process.c+process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+~process.b+process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a+~process.b+process.c)\n")
            s = Sequence(a+b+c+~d)
            self.assertEqual(s.copyAndExclude([a]).dumpPython(None),"cms.Sequence(process.b+process.c+~process.d)\n")
            self.assertEqual(s.copyAndExclude([b]).dumpPython(None),"cms.Sequence(process.a+process.c+~process.d)\n")
            self.assertEqual(s.copyAndExclude([c]).dumpPython(None),"cms.Sequence(process.a+process.b+~process.d)\n")
            self.assertEqual(s.copyAndExclude([d]).dumpPython(None),"cms.Sequence(process.a+process.b+process.c)\n")
            self.assertEqual(s.copyAndExclude([a,b,c,d]).dumpPython(None),"cms.Sequence()\n")

            e = DummyModule("e")
            f = DummyModule("f")
            g = DummyModule("g")
            h = DummyModule("h")
            t1 = Task(h)
            s = Sequence(a+b+c+~d, Task(e,f,Task(g,t1)))
            self.assertEqual(s.copyAndExclude([a,h]).dumpPython(None),"cms.Sequence(process.b+process.c+~process.d, cms.Task(process.e, process.f, process.g))\n")
            self.assertEqual(s.copyAndExclude([a,h]).dumpPython(None),"cms.Sequence(process.b+process.c+~process.d, cms.Task(process.e, process.f, process.g))\n")
            self.assertEqual(s.copyAndExclude([a,e,h]).dumpPython(None),"cms.Sequence(process.b+process.c+~process.d, cms.Task(process.f, process.g))\n")
            self.assertEqual(s.copyAndExclude([a,e,f,g,h]).dumpPython(None),"cms.Sequence(process.b+process.c+~process.d)\n")
            self.assertEqual(s.copyAndExclude([a,b,c,d]).dumpPython(None),"cms.Sequence(cms.Task(process.e, process.f, process.g, process.h))\n")
            self.assertEqual(s.copyAndExclude([t1]).dumpPython(None),"cms.Sequence(process.a+process.b+process.c+~process.d, cms.Task(process.e, process.f, process.g))\n")
            taskList = []
            taskVisitor = TaskVisitor(taskList)
            s.visit(taskVisitor)
            self.assertEqual(len(taskList),3)
            s2 = s.copyAndExclude([g,h])
            taskList[:] = []
            s2.visit(taskVisitor)
            self.assertEqual(len(taskList),1)
            t2 = Task(t1)
            taskList[:] = []
            t2.visit(taskVisitor)
            self.assertEqual(taskList[0],t1)
            s3 = Sequence(s)
            self.assertEqual(s3.copyAndExclude([a,h]).dumpPython(None),"cms.Sequence(process.b+process.c+~process.d, cms.Task(process.e, process.f, process.g))\n")
            s4 = Sequence(s)
            self.assertEqual(s4.copyAndExclude([a,b,c,d,e,f,g,h]).dumpPython(None),"cms.Sequence()\n")
            t1 = Task(e,f)
            t11 = Task(a)
            t11.setLabel("t11")
            t2 = Task(g,t1,h,t11)
            t3 = t2.copyAndExclude([e,h])
            self.assertTrue(t3.dumpPython(None) == "cms.Task(process.f, process.g, process.t11)\n")
            t4 = t2.copyAndExclude([e,f,g,h,a])
            self.assertTrue(t4.dumpPython(None) == "cms.Task()\n")
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
            p1 = Path(a+b+c)
            p2 = p1.copy()
            p1 += e
            self.assertEqual(p1.dumpPython(None),"cms.Path(process.a+process.b+process.c+process.e)\n")
            self.assertEqual(p2.dumpPython(None),"cms.Path(process.a+process.b+process.c)\n")
            t1 = Task(a, b)
            t2 = t1.copy()
            self.assertTrue(t1.dumpPython(None) == t2.dumpPython(None))
            t1Contents = list(t1._collection)
            t2Contents = list(t2._collection)
            self.assertTrue(id(t1Contents[0]) == id(t2Contents[0]))
            self.assertTrue(id(t1Contents[1]) == id(t2Contents[1]))
            self.assertTrue(id(t1._collection) != id(t2._collection))
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
            decoratedList = []
            lister = DecoratedNodeNameVisitor(decoratedList)
            p.visit(lister)
            ps = TestPSet()
            p.insertInto(ps,"p",decoratedList)
            self.assertEqual(ps._dict, {"p":vstring("a","b","c","d")})
            s = Sequence(b+c)
            p = Path(a+s+d)
            decoratedList[:] = []
            p.visit(lister)
            ps = TestPSet()
            p.insertInto(ps,"p",decoratedList)
            self.assertEqual(ps._dict, {"p":vstring("a","b","c","d")})
                        
    unittest.main()
