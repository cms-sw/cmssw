from Mixins import _ConfigureComponent, saveOrigin
from Mixins import _Unlabelable, _Labelable
from Mixins import _TypedParameterizable, _Parameterizable, PrintOptions
from SequenceTypes import _SequenceLeaf
from Types import vstring

from ExceptionHandling import *
class Service(_ConfigureComponent,_TypedParameterizable,_Unlabelable):
    def __init__(self,type_,*arg,**kargs):
        super(Service,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeService(self.type_(),self)
    def insertInto(self, processDesc):
        newpset = processDesc.newPSet()
        newpset.addString(True, "@service_type", self.type_())
        self.insertContentsInto(newpset)
        processDesc.addService(newpset)


class ESSource(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    def __init__(self,type_,*arg,**kargs):
        super(ESSource,self).__init__(type_,*arg,**kargs)
        saveOrigin(self, 1)
    def _placeImpl(self,name,proc):
        if name == '':
            name=self.type_()
        proc._placeESSource(name,self)
    def moduleLabel_(self,myname):
       result = myname
       if self.type_() == myname:
           result = ""
       return result
    def nameInProcessDesc_(self, myname):
       result = self.type_() + "@" + self.moduleLabel_(myname)
       return result


class ESProducer(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    def __init__(self,type_,*arg,**kargs):
        super(ESProducer,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        if name == '':
            name=self.type_()
        proc._placeESProducer(name,self)
    def moduleLabel_(self,myname):
       result = myname
       if self.type_() == myname:
           result = ''
       return result
    def nameInProcessDesc_(self, myname):
       result = self.type_() + "@" + self.moduleLabel_(myname)
       return result


class ESPrefer(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    """Used to set which EventSetup provider should provide a particular data item
    in the case where multiple providers are capable of delivering the data.
    The first argument specifies the C++ class type of the prodiver.
    If the provider has been given a label, you must specify that label as the second argument.
    Additional 'vstring' arguments maybe used to specify exactly which EventSetup Records
    are being preferred and optionally which data items within that Record.
    E.g.,
        #prefer all data in record 'OrangeRecord' from 'juicer'
        ESPrefer("ESJuicerProd", OrangeRecord=cms.vstring())
    or
        #prefer only "Orange" data in "OrangeRecord" from "juicer" 
        ESPrefer("ESJuicerProd", OrangeRecord=cms.vstring("ExtraPulp"))
    or 
        #prefer only "Orange" data with label "ExtraPulp" in "OrangeRecord" from "juicer" 
        ESPrefer("ESJuicerProd", OrangeRecord=cms.vstring("Orange/ExtraPulp"))
    """
    def __init__(self,type_,targetLabel='',*arg,**kargs):
        super(ESPrefer,self).__init__(type_,*arg,**kargs)
        self._targetLabel = targetLabel
        if targetLabel is None:
            self._targetLabel = str('')
        if kargs:
            for k,v in kargs.iteritems():
                if not isinstance(v,vstring):
                    raise RuntimeError('ESPrefer only allows vstring attributes. "'+k+'" is a '+str(type(v)))
    def _placeImpl(self,name,proc):
        proc._placeESPrefer(name,self)
    def nameInProcessDesc_(self, myname):
        # the C++ parser can give it a name like "label@prefer".  Get rid of that.
        return "esprefer_" + self.type_() + "@" + self._targetLabel
    def copy(self):
        returnValue = ESPrefer.__new__(type(self))
        returnValue.__init__(self.type_(), self._targetLabel)
        return returnValue
    def moduleLabel_(self, myname):
        return self._targetLabel
    def targetLabel_(self):
        return self._targetLabel
    def dumpPythonAs(self, label, options=PrintOptions()):
       result = options.indentation()
       basename = self._targetLabel
       if basename == '':
           basename = self.type_()
       if options.isCfg:
           # do either type or label
           result += 'process.prefer("'+basename+'"'
           if self.parameterNames_():
               result += ",\n"+_Parameterizable.dumpPython(self,options)+options.indentation()
           result +=')\n'
       else:
           # use the base class Module
           result += 'es_prefer_'+basename+' = cms.ESPrefer("'+self.type_()+'"'
           if self._targetLabel != '':
              result += ',"'+self._targetLabel+'"'
           if self.parameterNames_():
               result += ",\n"+_Parameterizable.dumpPython(self,options)+options.indentation()
           result += ')\n'
       return result

class _Module(_ConfigureComponent,_TypedParameterizable,_Labelable,_SequenceLeaf):
    """base class for classes which denote framework event based 'modules'"""
    __isStrict__ = False  
    def __init__(self,type_,*arg,**kargs):
        super(_Module,self).__init__(type_,*arg,**kargs)
        if _Module.__isStrict__:
            self.setIsFrozen()
        saveOrigin(self, 2)    
    def _clonesequence(self, lookuptable):
        try:
            return lookuptable[id(self)]
        except:
            # return something like "EDAnalyzer("foo", ...)"
            raise ModuleCloneError(self._errorstr())
    def _errorstr(self):
        typename = format_typename(self)
        return "%s('%s', ...)" %(typename, self.type_())

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
    def moduleLabel_(self,myname):
        return "@main_input"
    def nameInProcessDesc_(self,myname):
        return "@main_input"


class Looper(_ConfigureComponent,_TypedParameterizable):
    def __init__(self,type_,*arg,**kargs):
        super(Looper,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name,proc):
        proc._placeLooper(name,self)
    def moduleLabel_(self,myname):
        return "@main_looper"
    def nameInProcessDesc_(self, myname):
        return "@main_looper"



if __name__ == "__main__":
    import unittest
    from Types import *
    from SequenceTypes import *
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
            block = PSet(i = int32(9))
            m = EDProducer("DumbProducer", block, j = int32(10))
            self.assertEqual(9, m.i.value())
            self.assertEqual(10, m.j.value())
        def testESPrefer(self):
            juicer = ESPrefer("JuiceProducer")
            options = PrintOptions()
            options.isCfg = True
            self.assertEqual(juicer.dumpPythonAs("juicer", options), "process.prefer(\"JuiceProducer\")\n")
            options.isCfg = False
            self.assertEqual(juicer.dumpPythonAs("juicer", options), "es_prefer_JuiceProducer = cms.ESPrefer(\"JuiceProducer\")\n")

            juicer = ESPrefer("JuiceProducer","juicer")
            options = PrintOptions()
            options.isCfg = True
            self.assertEqual(juicer.dumpPythonAs("juicer", options), 'process.prefer("juicer")\n')
            options.isCfg = False
            self.assertEqual(juicer.dumpPythonAs("juicer", options), 'es_prefer_juicer = cms.ESPrefer("JuiceProducer","juicer")\n')
            juicer = ESPrefer("JuiceProducer",fooRcd=vstring())
            self.assertEqual(juicer.dumpConfig(options),
"""JuiceProducer { 
    vstring fooRcd = {
    }

}
""")
            options = PrintOptions()
            options.isCfg = True
            self.assertEqual(juicer.dumpPythonAs("juicer"),
"""process.prefer("JuiceProducer",
    fooRcd = cms.vstring()
)
""")
            options.isCfg = False
            self.assertEqual(juicer.dumpPythonAs("juicer", options),
"""es_prefer_JuiceProducer = cms.ESPrefer("JuiceProducer",
    fooRcd = cms.vstring()
)
""")
        
        def testService(self):
            empty = Service("Empty")
            withParam = Service("Parameterized",foo=untracked(int32(1)), bar = untracked(string("it")))
            self.assertEqual(withParam.foo.value(), 1)
            self.assertEqual(withParam.bar.value(), "it")
            self.assertEqual(empty.dumpPython(), "cms.Service(\"Empty\")\n")
            self.assertEqual(withParam.dumpPython(), "cms.Service(\"Parameterized\",\n    foo = cms.untracked.int32(1),\n    bar = cms.untracked.string(\'it\')\n)\n")
        def testSequences(self):
            m = EDProducer("MProducer")
            n = EDProducer("NProducer")
            m.setLabel("m")
            n.setLabel("n")
            s1 = Sequence(m*n)
            options = PrintOptions()
            print s1.dumpPython(options)



    unittest.main()
