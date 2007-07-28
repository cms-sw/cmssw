from Mixins import _ConfigureComponent
from Mixins import _Unlabelable, _Labelable
from Mixins import _TypedParameterizable 
from SequenceTypes import _Sequenceable

from ExceptionHandling import *
import libFWCoreParameterSet

class Service(_ConfigureComponent,_TypedParameterizable,_Unlabelable):
    def __init__(self,type_,*arg,**kargs):
        super(Service,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeService(self.type_(),self)


class ESSource(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    def __init__(self,type_,*arg,**kargs):
        super(ESSource,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        if name == '':
            name=self.type_()
        proc._placeESSource(name,self)


class ESProducer(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    def __init__(self,type_,*arg,**kargs):
        super(ESProducer,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        if name == '':
            name=self.type_()
        proc._placeESProducer(name,self)



class ESPrefer(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    def __init__(self,type_,*arg,**kargs):
        super(ESPrefer,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        if name == '':
            name=self.type_()
        proc._placeESPrefer(name,self)


class _Module(_ConfigureComponent,_TypedParameterizable,_Labelable,_Sequenceable):
    """base class for classes which denote framework event based 'modules'"""
    def __init__(self,type_,*arg,**kargs):
        super(_Module,self).__init__(type_,*arg,**kargs)
    def _clonesequence(self, lookuptable):
        try:
            return lookuptable[id(self)]
        except:
            # return something like "EDAnalyzer("foo", ...)"
            raise ModuleCloneError(self._errorstr())
    def _errorstr(self):
        typename = format_typename(self)
        return "%s('%s', ...)" %(typename, self.type_())
    def insertInto(self, parameterSet, myname):
        newpset = libFWCoreParameterSet.ParameterSet()
        newpset.addString(True, "@module_label", myname)
        newpset.addString(True, "@module_type", self.type_())
        self.insertContentsInto(newpset)
        parameterSet.addPSet(True, myname, newpset)
        

class EDProducer(_Module):
    def __init__(self,type_,*arg,**kargs):
        super(EDProducer,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeProducer(name,self)


class EDFilter(_Module):
    def __init__(self,type_,*arg,**kargs):
        super(EDFilter,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeFilter(name,self)


class EDAnalyzer(_Module):
    def __init__(self,type_,*arg,**kargs):
        super(EDAnalyzer,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeAnalyzer(name,self)


class OutputModule(_Module):
    def __init__(self,type_,*arg,**kargs):
        super(OutputModule,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeOutputModule(name,self)


class Source(_ConfigureComponent,_TypedParameterizable):
    def __init__(self,type_,*arg,**kargs):
        super(Source,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeSource(name,self)


class Looper(_ConfigureComponent,_TypedParameterizable):
    def __init__(self,type_,*arg,**kargs):
        super(Looper,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeLooper(name,self)



if __name__ == "__main__":
    import unittest
    from Types import *
    class TestModules(unittest.TestCase):
        def testEDAnalyzer(self):
            empty = EDAnalyzer("Empty")
            withParam = EDAnalyzer("Parameterized",foo=untracked(int32(1)), bar = untracked(string("it")))
            self.assertEqual(withParam.foo.value(), 1)
            self.assertEqual(withParam.bar.value(), "it")
            aCopy = withParam.copy()
            self.assertEqual(aCopy.foo.value(), 1)
            self.assertEqual(aCopy.bar.value(), "it")
            withType = EDAnalyzer("Test",type = int32(1))
            self.assertEqual(withType.type.value(),1)
        
        def testService(self):
            empty = Service("Empty")
            withParam = Service("Parameterized",foo=untracked(int32(1)), bar = untracked(string("it")))
            self.assertEqual(withParam.foo.value(), 1)
            self.assertEqual(withParam.bar.value(), "it")

    unittest.main()
