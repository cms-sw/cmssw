from Mixins import _ConfigureComponent
from Mixins import _Unlabelable, _Labelable
from Mixins import _TypedParameterizable 
from SequenceTypes import _Sequenceable

class ModuleCloneError(Exception):
    pass 


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


class ESPrefer(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    def __init__(self,type,*arg,**kargs):
        super(ESPrefer,self).__init__(type,*arg,**kargs)
    def _placeImpl(self,name,proc):
        if name == '':
            name=self.type_()
        proc._placeESPrefer(name,self)


class _Module(_ConfigureComponent,_TypedParameterizable,_Labelable,_Sequenceable):
    """base class for classes which denote framework event based 'modules'"""
    def __init__(self,type,*arg,**kargs):
        super(_Module,self).__init__(type,*arg,**kargs)
    def _clonesequence(self, lookuptable):
        try:
            return lookuptable[id(self)]
        except:
            typename = str(type(self)).split("'")[1].split(".")[1] # has to be improved!
            # return something like "EDAnalyzer("foo", ...)"
            raise ModuleCloneError("%s('%s', ...)" %(typename, self.type_()))

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
        
        def testService(self):
            empty = Service("Empty")
            withParam = Service("Parameterized",foo=untracked(int32(1)), bar = untracked(string("it")))
            self.assertEqual(withParam.foo.value(), 1)
            self.assertEqual(withParam.bar.value(), "it")

    unittest.main()
