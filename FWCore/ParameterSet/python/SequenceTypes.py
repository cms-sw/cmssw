
from Mixins import _ConfigureComponent
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
    def fillModulesList(self, l):
        # hope everything put into the process has a label
        l.add(self.label())
    def findHardDependencies(self, sequenceName, dependencyDict):
        pass

class _ModuleSequenceType(_ConfigureComponent, _Labelable):
    """Base class for classes which define a sequence of modules"""
    def __init__(self,*arg, **argv):
        self.__dict__["_isFrozen"] = False
        if len(arg) != 1:
            typename = format_typename(self)
            msg = format_outerframe(2) 
            msg += "%s takes exactly one input value. But the following ones are given:\n" %typename
            for item,i in zip(arg, xrange(1,20)):
                msg += "    %i) %s \n"  %(i, item._errorstr())
            msg += "Maybe you forgot to combine them via '*' or '+'."     
            raise TypeError(msg)
        self._checkIfSequenceable(arg[0])
        self._seq = arg[0]
        self._isModified = False
    def isFrozen(self):
        return self._isFrozen
    def setIsFrozen(self):
        self._isFrozen = True 
    def _place(self,name,proc):
        self._placeImpl(name,proc)
    def __imul__(self,rhs):
        self._checkIfSequenceable(rhs)
        self._seq = _SequenceOpAids(self._seq,rhs)
        return self
    def __iadd__(self,rhs):
        self._checkIfSequenceable(rhs)
        self._seq = _SequenceOpFollows(self._seq,rhs)
        return self
    def _checkIfSequenceable(self,v):
        if not isinstance(v,_Sequenceable):
            typename = format_typename(self)
            msg = format_outerframe(2)
            msg += "%s only takes arguments of types which are allowed in a sequence, but was given:\n" %typename
            msg +=format_typename(v)
            msg +="\nPlease remove the problematic object from the argument list"
            raise TypeError(msg)
    def __str__(self):
        return str(self._seq)
    def dumpConfig(self, options):
        return '{'+self._seq.dumpSequenceConfig()+'}\n'
    def dumpPython(self, options):
        return 'cms.'+type(self).__name__+'('+self._seq.dumpSequencePython()+')\n'
    def __repr__(self):
        return "cms."+type(self).__name__+'('+str(self._seq)+')\n'
    def copy(self):
        returnValue =_ModuleSequenceType.__new__(type(self))
        returnValue.__init__(self._seq)
        return returnValue
    def _postProcessFixup(self,lookuptable):
        self._seq = self._seq._clonesequence(lookuptable)
        return self
    def replace(self, original, replacement):
        if not isinstance(original,_Sequenceable) or not isinstance(replacement,_Sequenceable):
           raise ValueError
        else:
           self._replace(original, replacement)
    def _replace(self, original, replacement):
        if self._seq == original:
            self._seq = replacement
        else:
            self._seq._replace(original,replacement)
    def resolve(self, processDict):
        self._seq = self._seq.resolve(processDict)
        return self
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
    def nameInProcessDesc_(self, myname):
        return myname
    def fillNamesList(self, l, processDict):
        return self._seq.fillNamesList(l, processDict)
    def insertInto(self, parameterSet, myname, processDict):
        # represented just as a list of names in the ParameterSet
        l = []
        self.resolve(processDict)
        self.fillNamesList(l, processDict)
        parameterSet.addVString(True, myname, l)
    def visit(self,visitor):
        """Passes to visitor's 'enter' and 'leave' method each item describing the module sequence.
        If the item contains 'sub' items then visitor will see those 'sub' items between the
        item's 'enter' and 'leave' calls.
        """
        self._seq.visitNode(visitor)
    def fillModulesList(self, l):
        self._seq.fillModulesList(l)
    def findHardDependencies(self, sequenceName, dependencyDict):
        self._seq.findHardDependencies(self.label(), dependencyDict)


class _SequenceOperator(_Sequenceable):
    """Used in the expression tree for a sequence"""
    def __init__(self, left, right):
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
    def resolve(self, processDict):
        self._left = self._left.resolve(processDict)
        self._right = self._right.resolve(processDict)
        return self
    def fillNamesList(self, l, processDict):
        self._left.fillNamesList(l, processDict)
        self._right.fillNamesList(l, processDict)
    def isOperation(self):
        return True
    def _visitSubNodes(self,visitor):
        self._left.visitNode(visitor)
        self._right.visitNode(visitor)
    def _precedence(self):
        """Precedence order for this operation, the larger the value the higher the precedence"""
        raise RuntimeError("_precedence must be overwritten by inheriting classes")
        return 0
    def fillModulesList(self, l):
        self._left.fillModulesList(l)
        self._right.fillModulesList(l)


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
        self._right.fillModulesList(rhs)
        lhs = set()
        self._left.fillModulesList(lhs)
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
    def _findDependencies(self,knownDeps, presentDeps):
        self._operand._findDependencies(knownDeps, presentDeps)
    def fillNamesList(self, l, processDict):
        l.append(self.dumpSequenceConfig())
    def _clonesequence(self, lookuptable):
        return type(self)(self._operand._clonesequence(lookuptable))
    def _replace(self, original, replacement):
        if self._operand == original:
            self._operand = replacement
        else:
            self._operand._replace(original, replacement)
    def resolve(self, processDict):
        self._operand = self._operand.resolve(processDict)
        return self
    def isOperation(self):
        return True
    def _visitSubNodes(self,visitor):
        self._operand.visitNode(visitor)
    def fillModulesList(self, l):
        self._operand.fillModulesList(l)
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
            clone = type(self)(self._seq._clonesequence(lookuptable))
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
    def fillNamesList(self, l, processDict):
        """ Resolves SequencePlaceholders """
        if not self._name in processDict:
            raise RuntimeError("The SequencePlaceholder "+self._name+ " cannot be resolved")
        else:
            processDict[self._name].fillNamesList(l, processDict)
    def copy(self):
        returnValue =SequencePlaceholder.__new__(type(self))
        returnValue.__init__(self._name)
        return returnValue
    def dumpSequenceConfig(self):
        return self._name
    def dumpSequencePython(self):
        return "process."+self._name
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
    def fillNamesList(self, l, processDict):
        for seq in self:
            seq.fillNamesList(l, processDict)
    def enforceDependencies(self):
        # I don't think we need the processDict
        processDict = dict()
        dependencyDict = dict()
        names = list()
        ok = True
        errors = list()
        for seq in self:
            seq.fillNamesList(names, processDict) 
            seq.findHardDependencies('schedule', dependencyDict)
        for label, deps in dependencyDict.iteritems():
            # see if it's in 
            try:
                thisPos = names.index(label)
                # we had better find all the dependencies
                for dep in deps.depSet:
                    if names[0:thisPos].count(dep) == 0:
                        ok = False 
                        message = "WARNING:"+label+" depends on "+dep+", as declared in " \
                                  + deps.sequenceName+", but not found in schedule"
                        print message
            except:  
                # can't find it?  No big deal.
                pass


if __name__=="__main__":
    import unittest
    class DummyModule(_Labelable, _Sequenceable):
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
            d = dict()
            p.fillNamesList(l, d)
            self.assertEqual(l, ['a', 'b'])
            l = list()
            p5.fillNamesList(l, d)
            self.assertEqual(l, ['a', '-b'])
            l = list()
            p7.fillNamesList(l, d)
            self.assertEqual(l, ['a', '!b'])

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
            p.fillNamesList(l, d)
            self.assertEqual(l, ['m1', 'm2'])
            p.resolve(d)
            l = list()
            p.fillNamesList(l, d)
            self.assertEqual(l, ['m1', 'm2'])
        def testReplace(self):
            m1 = DummyModule("m1")
            m2 = DummyModule("m2")
            m3 = DummyModule("m3")
            m4 = DummyModule("m4")
            m5 = DummyModule("m5")
 
            s1 = Sequence(m1*~m2*m1*m2*ignore(m2))
            s2 = Sequence(m1*m2)
            s3 = Sequence(~m1*s2)  
            d = {'m1':m1 ,'m2':m2, 'm3':m3,'s1':s1, 's2':s2}  
            l = []
            s1.fillNamesList(l,d)
            self.assertEqual(l,['m1', '!m2', 'm1', 'm2', '-m2'])
            s1.replace(m2,m3)
            l = []
            s1.fillNamesList(l,d)
            self.assertEqual(l,['m1', '!m3', 'm1', 'm3', '-m3'])
            s2 = Sequence(m1*m2)
            s3 = Sequence(~m1*s2)
            s3.fillNamesList(l,d)
            self.assertEqual(l,['m1', '!m3', 'm1', 'm3', '-m3', '!m1', 'm1', 'm2'])
            l= []
            s3.replace(s2,m1)
            s3.fillNamesList(l,d)
            self.assertEqual(l,['!m1', 'm1'])
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
            self.assertEqual(deps['m2'].depSet, set(['m1']))
            self.assertEqual(deps['m3'].depSet, set(['m1']))
            self.assertEqual(deps['m2'].sequenceName, 's4')
            self.assertEqual(deps['m3'].sequenceName, 's4')
            self.failIf(deps.has_key('m4'))
            self.failIf(deps.has_key('m1'))
            deps = dict()
            p5 = Path(s4*m5)
            p5.setLabel('p5')
            p5.findHardDependencies('top',deps)
            self.assertEqual(len(deps['m5'].depSet), 4)
            self.assertEqual(deps['m5'].sequenceName, 'p5')
            self.assertEqual(deps['m3'].sequenceName, 's4')


    unittest.main()
                          


                           
    

        
        
