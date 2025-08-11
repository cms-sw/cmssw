from .Mixins import _ConfigureComponent, saveOrigin
from .Mixins import _Unlabelable, _Labelable
from .Mixins import _TypedParameterizable, _Parameterizable, PrintOptions, specialImportRegistry
from .SequenceTypes import _SequenceLeaf
from .Types import vstring, EDAlias


import copy
from .ExceptionHandling import *
class Service(_ConfigureComponent,_TypedParameterizable,_Unlabelable):
    def __init__(self,type_,*arg,**kargs):
        super(Service,self).__init__(type_,*arg,**kargs)
        self._inProcess = False
    def _placeImpl(self,name:str,proc):
        self._inProcess = True
        proc._placeService(self.type_(),self)
    def insertInto(self, processDesc):
        newpset = processDesc.newPSet()
        newpset.addString(True, "@service_type", self.type_())
        self.insertContentsInto(newpset)
        processDesc.addService(newpset)
    def dumpSequencePython(self, options:PrintOptions=PrintOptions()) -> str:
        return "process." + self.type_()
    def _isTaskComponent(self) -> bool:
        return True
    def isLeaf(self) -> bool:
        return True
    def __str__(self):
        return str(self.type_())

class ESSource(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    def __init__(self,type_,*arg,**kargs):
        super(ESSource,self).__init__(type_,*arg,**kargs)
        saveOrigin(self, 1)
    def _placeImpl(self,name:str,proc):
        if name == '':
            name=self.type_()
        proc._placeESSource(name,self)
    def moduleLabel_(self,myname:str) -> str:
       result = myname
       if self.type_() == myname:
           result = ""
       return result
    def nameInProcessDesc_(self, myname:str) -> str:
       result = self.type_() + "@" + self.moduleLabel_(myname)
       return result
    def _isTaskComponent(self) -> bool:
        return True
    def isLeaf(self) -> bool:
        return True

class ESProducer(_ConfigureComponent,_TypedParameterizable,_Unlabelable,_Labelable):
    def __init__(self,type_,*arg,**kargs):
        super(ESProducer,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name:str,proc):
        if name == '':
            name=self.type_()
        proc._placeESProducer(name,self)
    def moduleLabel_(self,myname:str) -> str:
       result = myname
       if self.type_() == myname:
           result = ''
       return result
    def nameInProcessDesc_(self, myname:str) -> str:
       result = self.type_() + "@" + self.moduleLabel_(myname)
       return result
    def _isTaskComponent(self) -> bool:
        return True
    def isLeaf(self) -> bool:
        return True

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
    def __init__(self,type_,targetLabel:str='',*arg,**kargs):
        super(ESPrefer,self).__init__(type_,*arg,**kargs)
        self._targetLabel = targetLabel
        if targetLabel is None:
            self._targetLabel = str('')
        if kargs:
            for k,v in kargs.items():
                if not isinstance(v,vstring):
                    raise RuntimeError('ESPrefer only allows vstring attributes. "'+k+'" is a '+str(type(v)))
    def _placeImpl(self,name:str,proc):
        proc._placeESPrefer(name,self)
    def nameInProcessDesc_(self, myname:str) -> str:
        # the C++ parser can give it a name like "label@prefer".  Get rid of that.
        return "esprefer_" + self.type_() + "@" + self._targetLabel
    def copy(self):
        returnValue = ESPrefer.__new__(type(self))
        returnValue.__init__(self.type_(), self._targetLabel)
        return returnValue
    def moduleLabel_(self, myname:str) -> str:
        return self._targetLabel
    def targetLabel_(self) -> str:
        return self._targetLabel
    def dumpPythonAs(self, label, options:PrintOptions=PrintOptions()) -> str:
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
            raise ModuleCloneError(self._errorstr())
    def _errorstr(self):
         # return something like "EDAnalyzer("foo", ...)"
        typename = format_typename(self)
        return "%s('%s', ...)" %(typename, self.type_())
    
    def setPrerequisites(self, *libs):
        self.__dict__["libraries_"] = libs

    def insertInto(self, parameterSet, myname:str):
        if "libraries_" in self.__dict__:
            from ctypes import LibraryLoader, CDLL
            import platform
            loader = LibraryLoader(CDLL)
            ext = platform.uname()[0] == "Darwin" and "dylib" or "so"
            [loader.LoadLibrary("lib%s.%s" % (l, ext)) for l in self.libraries_]
        super(_Module,self).insertInto(parameterSet,myname)

class EDProducer(_Module):
    def __init__(self,type_,*arg,**kargs):
        super(EDProducer,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name:str,proc):
        proc._placeProducer(name,self)
    def _isTaskComponent(self):
        return True

class EDFilter(_Module):
    def __init__(self,type_,*arg,**kargs):
        super(EDFilter,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name:str,proc):
        proc._placeFilter(name,self)
    def _isTaskComponent(self):
        return True

class EDAnalyzer(_Module):
    def __init__(self,type_,*arg,**kargs):
        super(EDAnalyzer,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name:str,proc):
        proc._placeAnalyzer(name,self)


class OutputModule(_Module):
    def __init__(self,type_,*arg,**kargs):
        super(OutputModule,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name:str,proc):
        proc._placeOutputModule(name,self)


class Source(_ConfigureComponent,_TypedParameterizable):
    def __init__(self,type_,*arg,**kargs):
        super(Source,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name:str,proc):
        proc._placeSource(name,self)
    def moduleLabel_(self,myname:str):
        return "@main_input"
    def nameInProcessDesc_(self,myname:str):
        return "@main_input"


class Looper(_ConfigureComponent,_TypedParameterizable):
    def __init__(self,type_,*arg,**kargs):
        super(Looper,self).__init__(type_,*arg,**kargs)
    def _placeImpl(self,name:str,proc):
        proc._placeLooper(name,self)
    def moduleLabel_(self,myname:str):
        return "@main_looper"
    def nameInProcessDesc_(self, myname:str):
        return "@main_looper"


# Need to be a module-level function for the configuration with a
# SwitchProducer to be pickleable.
def _switch_cpu(accelerators):
    return (True, 1)

class SwitchProducer(EDProducer):
    """This purpose class is to provide a switch of EDProducers for a single module/product label.

    The decision is done at the time when the python configuration is
    translated to C++. This class is generic, and intended to be
    inherited for concrete switches. Example:

    class SwitchProducerFoo(SwitchProducer):
        def __init__(self, **kargs):
            super(SwitchProducerFoo,self).__init__(
                dict(case1 = case1Func, case2 = case2Func),
                **kargs
            )

    foo = SwitchProducerFoo(
        case1 = EDProducer("Producer1"),
        case2 = EDProducer("Producer2")
    )

    Here case1Func and case2Func are functions that return a (bool,
    int) tuple, where the bool tells whether that case is enabled or
    not, and the int tells the priority of that case. The case with
    the highest priority among those that are enabled will get chosen.

    The end result is that the product(s) labeled as "foo" will be
    produced with one of the producers. It would be good if their
    output product types and instance names would be the same (or very
    close).
    """
    def __init__(self, caseFunctionDict, **kargs):
        super(SwitchProducer,self).__init__(None)
        self._caseFunctionDict = copy.copy(caseFunctionDict)
        self.__setParameters(kargs)
        self._isModified = False

    def setLabel(self, label:str):
        super().setLabel(label)
        # SwitchProducer owns the contained modules, and therefore
        # need to set / unset the label for them explicitly here
        for case in self.parameterNames_():
            producer = self.__dict__[case]
            producer.setLabel(self.caseLabel_(label, case) if label is not None else None)

    @staticmethod
    def getCpu():
        """Returns a function that returns the priority for a CPU "computing device". Intended to be used by deriving classes."""
        return _switch_cpu

    def _chooseCase(self, accelerators):
        """Returns the name of the chosen case."""
        cases = self.parameterNames_()
        bestCase = None
        for case in cases:
            (enabled, priority) = self._caseFunctionDict[case](accelerators)
            if enabled and (bestCase is None or bestCase[0] < priority):
                bestCase = (priority, case)
        if bestCase is None:
            raise RuntimeError("All cases '%s' were disabled" % (str(cases)))
        return bestCase[1]

    def _getProducer(self, accelerators):
        """Returns the EDroducer of the chosen case"""
        return self.__dict__[self._chooseCase(accelerators)]

    @staticmethod
    def __typeIsValid(typ) -> bool:
        return (isinstance(typ, EDProducer) and not isinstance(typ, SwitchProducer)) or isinstance(typ, EDAlias)

    def __addParameter(self, name:str, value):
        if not self.__typeIsValid(value):
            raise TypeError(name+" does not already exist, so it can only be set to a cms.EDProducer or cms.EDAlias")
        if name not in self._caseFunctionDict:
            raise ValueError("Case '%s' is not allowed (allowed ones are %s)" % (name, ",".join(self._caseFunctionDict.keys())))
        if name in self.__dict__:
            message = "Duplicate insert of member " + name
            message += "\nThe original parameters are:\n"
            message += self.dumpPython() + '\n'
            raise ValueError(message)
        self.__dict__[name]=value
        if self.hasLabel_():
            value.setLabel(self.caseLabel_(self.label_(), name))
        self._Parameterizable__parameterNames.append(name)
        self._isModified = True

    def __setParameters(self, parameters):
        for name, value in parameters.items():
            self.__addParameter(name, value)

    def __setattr__(self, name:str, value):
        # Following snippet copied and customized from
        # _Parameterizable in order to support Modifier.toModify
        #
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
            super(SwitchProducer, self).__setattr__(name,value)
        elif not name in self.__dict__:
            self.__addParameter(name, value)
            self._isModified = True
        else:
            if not self.__typeIsValid(value):
                raise TypeError(name+" can only be set to a cms.EDProducer or cms.EDAlias")
            # We should always receive an cms.EDProducer
            self.__dict__[name] = value
            if self.hasLabel_():
                value.setLabel(self.caseLabel_(self.label_(), name))
            self._isModified = True

    def clone(self, **params):
        returnValue = SwitchProducer.__new__(type(self))

        # Need special treatment as cms.EDProducer is not a valid parameter type (except in this case)
        myparams = dict()
        for name, value in params.items():
            if value is None:
                continue
            elif isinstance(value, dict):
                myparams[name] = self.__dict__[name].clone(**value)
            else: # value is an EDProducer
                myparams[name] = value.clone()

        # Add the ones that were not customized
        for name in self.parameterNames_():
            if name not in params:
                myparams[name] = self.__dict__[name].clone()
        returnValue.__init__(**myparams)
        returnValue._isModified = False
        returnValue._isFrozen = False
        saveOrigin(returnValue, 1)
        return returnValue

    def dumpPython(self, options:PrintOptions=PrintOptions()) -> str:
        # Note that if anyone uses the generic SwitchProducer instead
        # of a derived-one, the information on the functions for the
        # producer decision is lost
        specialImportRegistry.registerUse(self)
        result = "%s(" % self.__class__.__name__ # not including cms. since the deriving classes are not in cms "namespace"
        options.indent()
        for resource in sorted(self.parameterNames_()):
            result += "\n" + options.indentation() + resource + " = " + getattr(self, resource).dumpPython(options).rstrip() + ","
        if result[-1] == ",":
            result = result.rstrip(",")
        options.unindent()
        result += "\n)\n"
        return result

    def directDependencies(self):
        # XXX FIXME handle SwitchProducer dependencies
        return []

    def nameInProcessDesc_(self, myname:str):
        return myname
    def moduleLabel_(self, myname:str):
        return myname
    def caseLabel_(self, name:str, case:str):
        return name+"@"+case
    def cloneCaseModuleIfNeeded_(self, myname:str, case:str):
        mod = self.__dict__[case]
        if (self.caseLabel_(myname, case) != mod.label_()):
            clne = mod.clone()
            clne.setLabel(None)
            clne.setLabel(self.caseLabel_(myname, case))
            self.__dict__[case] = clne
            mod = clne
        return mod
    def modulesForConditionalTask_(self):
        # Need the contained modules (not EDAliases) for ConditionalTask
        ret = []
        for case in self.parameterNames_():
            caseobj = self.__dict__[case]
            if not isinstance(caseobj, EDAlias):
                ret.append(self.cloneCaseModuleIfNeeded_(self.label_(), case))
        return ret
    def appendToProcessDescLists_(self, modules, aliases, myname:str):
        # This way we can insert the chosen EDProducer to @all_modules
        # so that we get easily a worker for it
        modules.append(myname)
        for case in self.parameterNames_():
            if isinstance(self.__dict__[case], EDAlias):
                aliases.append(self.caseLabel_(myname, case))
            else:
                modules.append(self.caseLabel_(myname, case))

    def insertInto(self, parameterSet, myname:str, accelerators):
        for case in self.parameterNames_():
            producer = self.__dict__[case]
            if isinstance(producer, EDProducer):
                producer = self.cloneCaseModuleIfNeeded_(myname, case)
            producer.insertInto(parameterSet, self.caseLabel_(myname, case))
        newpset = parameterSet.newPSet()
        newpset.addString(True, "@module_label", self.moduleLabel_(myname))
        newpset.addString(True, "@module_type", "SwitchProducer")
        newpset.addString(True, "@module_edm_type", "EDProducer")
        newpset.addVString(True, "@all_cases", [self.caseLabel_(myname, p) for p in self.parameterNames_()])
        newpset.addString(False, "@chosen_case", self.caseLabel_(myname, self._chooseCase(accelerators)))
        parameterSet.addPSet(True, self.nameInProcessDesc_(myname), newpset)

    def _placeImpl(self,name:str,proc):
        proc._placeSwitchProducer(name,self)
#        for case in self.parameterNames_():
#            caseLabel = self.caseLabel_(name, case)
#            caseObj = self.__dict__[case]
#
#            if isinstance(caseObj, EDAlias):
#                # EDAliases end up in @all_aliases automatically
#                proc._placeAlias(caseLabel, caseObj)
#            else:
#                # Note that these don't end up in @all_modules
#                # automatically because they're not part of any
#                # Task/Sequence/Path
#                proc._placeProducer(caseLabel, caseObj)

    def _clonesequence(self, lookuptable):
        try:
            return lookuptable[id(self)]
        except:
            raise ModuleCloneError(self._errorstr())
    def _errorstr(self):
        return "SwitchProducer"


if __name__ == "__main__":
    import unittest
    from .Types import *
    from .SequenceTypes import *

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
            self.assertEqual(withParam.dumpPython(), "cms.Service(\"Parameterized\",\n    bar = cms.untracked.string(\'it\'),\n    foo = cms.untracked.int32(1)\n)\n")
        def testSequences(self):
            m = EDProducer("MProducer")
            n = EDProducer("NProducer")
            m.setLabel("m")
            n.setLabel("n")
            s1 = Sequence(m*n)
            options = PrintOptions()

        def testIsTaskComponent(self):
            m = EDProducer("x")
            self.assertTrue(m._isTaskComponent())
            self.assertTrue(m.isLeaf())
            m = EDFilter("x")
            self.assertTrue(m._isTaskComponent())
            self.assertTrue(m.isLeaf())
            m = OutputModule("x")
            self.assertFalse(m._isTaskComponent())
            self.assertTrue(m.isLeaf())
            m = EDAnalyzer("x")
            self.assertFalse(m._isTaskComponent())
            self.assertTrue(m.isLeaf())
            m = Service("x")
            self.assertTrue(m._isTaskComponent())
            self.assertTrue(m.isLeaf())
            m = ESProducer("x")
            self.assertTrue(m._isTaskComponent())
            self.assertTrue(m.isLeaf())
            m = ESSource("x")
            self.assertTrue(m._isTaskComponent())
            self.assertTrue(m.isLeaf())
            m = Sequence()
            self.assertFalse(m._isTaskComponent())
            self.assertFalse(m.isLeaf())
            m = Path()
            self.assertFalse(m._isTaskComponent())
            m = EndPath()
            self.assertFalse(m._isTaskComponent())
            m = Task()
            self.assertTrue(m._isTaskComponent())
            self.assertFalse(m.isLeaf())

    unittest.main()
