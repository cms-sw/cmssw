
from Mixins import _ConfigureComponent
from Mixins import _Labelable, _Unlabelable
from Mixins import _ValidatingListBase
from ExceptionHandling import *

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

class _ModuleSequenceType(_ConfigureComponent, _Labelable):
    """Base class for classes which define a sequence of modules"""
    def __init__(self,*arg, **argv):
        if len(arg) != 1:
            typename = format_typename(self)
            msg = format_outerframe(2) 
            msg += "%s takes exactly one input value. But the following ones are given:\n" %typename
            for item,i in zip(arg, xrange(1,20)):
                msg += "    %i) %s \n"  %(i, item._errorstr())
            msg += "Maybe you forgot to combine them via '*' or '+'."     
            raise TypeError(msg)
        if not isinstance(arg[0],_Sequenceable):
            typename = format_typename(self)
            msg = format_outerframe(2)
            msg += "%s only takes arguments of types which are allowed in a sequence, but was given:\n" %typename
            msg +=format_typename(arg[0])
            msg +="\nPlease remove the problematic object from the argument list"
            raise TypeError(msg)

        self._seq = arg[0]
    def _place(self,name,proc):
        self._placeImpl(name,proc)
    def __imul__(self,rhs):
        if not isinstance(rhs,_Sequenceable):
            typename = format_typename(self)
            msg = format_outerframe(2)
            msg += "%s only takes arguments of types which are allowed in a sequence, but was given:\n" %typename
            msg +=format_typename(rhs)
            msg +="\nPlease remove the problematic object from the argument list"
            raise TypeError(msg)
        self._seq = _SequenceOpAids(self._seq,rhs)
        return self
    def __iadd__(self,rhs):
        if not isinstance(rhs,_Sequenceable):
            typename = format_typename(self)
            msg = format_outerframe(2)
            msg += "%s only takes arguments of types which are allowed in a sequence, but was given:\n" %typename
            msg +=format_typename(rhs)
            msg +="\nPlease remove the problematic object from the argument list"
            raise TypeError(msg)
        self._seq = _SequenceOpFollows(self._seq,rhs)
        return self
    def __str__(self):
        return str(self._seq)
    def dumpConfig(self,indent,deltaIndent):
        return '{'+self._seq.dumpSequenceConfig()+'}\n'
    def dumpPython(self,indent,deltaIndent):
        return '('+self._seq.dumpSequencePython()+')\n'
    def copy(self):
        returnValue =_ModuleSequenceType.__new__(type(self))
        returnValue.__init__(self._seq)
        return returnValue
    def _postProcessFixup(self,lookuptable):
        self._seq = self._seq._clonesequence(lookuptable)
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
    def insertInto(self, parameterSet, myname):
        # represented just as a list of names in the ParameterSet
        l = []
        self._seq.fillNamesList(l)
        parameterSet.addVString(True, myname, l)

class _SequenceOpAids(_Sequenceable):
    """Used in the expression tree for a sequence as a stand in for the ',' operator"""
    def __init__(self, left, right):
        self.__left = left
        self.__right = right
    def __str__(self):
        return '('+str(self.__left)+'*'+str(self.__right) +')'
    def dumpSequenceConfig(self):
        return '('+self.__left.dumpSequenceConfig()+','+self.__right.dumpSequenceConfig()+')'
    def dumpSequencePython(self):
        return '('++self.__left.dumpSequencePython()+','+self.__right.dumpSequencePython()+')'
    def _findDependencies(self,knownDeps,presentDeps):
        #do left first and then right since right depends on left
        self.__left._findDependencies(knownDeps,presentDeps)
        self.__right._findDependencies(knownDeps,presentDeps)
    def _clonesequence(self, lookuptable):
        return type(self)(self.__left._clonesequence(lookuptable),self.__right._clonesequence(lookuptable))
    def fillNamesList(self, l):
        self.__left.fillNamesList(l)
        self.__right.fillNamesList(l)


class _SequenceNegation(_Sequenceable):
    """Used in the expression tree for a sequence as a stand in for the '!' operator"""
    def __init__(self, operand):
        self.__operand = operand
    def __str__(self):
        return '!%s' %self.__operand
    def dumpSequenceConfig(self):
        return '!%s' %self.__operand.dumpSequenceConfig()
    def dumpSequencePython(self):
        return '!%s' %self.__operand.dumpSequencePython()
    def _findDependencies(self,knownDeps, presentDeps):
        self.__operand._findDependencies(knownDeps, presentDeps)
    def fillNamesList(self, l):
        l.append(self.__str__())


class _SequenceOpFollows(_Sequenceable):
    """Used in the expression tree for a sequence as a stand in for the '&' operator"""
    def __init__(self, left, right):
        self.__left = left
        self.__right = right
    def __str__(self):
        return '('+str(self.__left)+'+'+str(self.__right) +')'
    def dumpSequenceConfig(self):
        return '('+self.__left.dumpSequenceConfig()+'&'+self.__right.dumpSequenceConfig()+')'
    def dumpSequencePython(self):
        return '('+self.__left.dumpSequencePython()+'&'+self.__right.dumpSequencePython()+')'
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
    def fillNamesList(self, l):
        self.__left.fillNamesList(l)
        self.__right.fillNamesList(l)



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
