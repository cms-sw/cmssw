import FWCore.ParameterSet.Config as cms

class ExternalGeneratorFilter(cms.EDFilter):
    def __init__(self, prod, _external_process_waitTime_ = cms.untracked.uint32(60), _external_process_verbose_ = cms.untracked.bool(False),
                 _external_process_components_ = cms.vstring()):
        self.__dict__['_external_process_verbose_']=_external_process_verbose_
        self.__dict__['_external_process_waitTime_']=_external_process_waitTime_
        self.__dict__['_external_process_components_'] = _external_process_components_
        self.__dict__['_prod'] = prod
        super(cms.EDFilter,self).__init__('ExternalGeneratorFilter')
    def __setattr__(self, name, value):
        if name =='_external_process_verbose_':
            return self.__dict__['_external_process_verbose_']
        setattr(self._prod, name, value)
    def __getattr__(self, name):
        if name =='_prod':
            return self.__dict__['_prod']
        if name == '_external_process_verbose_':
            return self.__dict__['_external_process_verbose_']
        return getattr(self._prod, name)
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def clone(self, **params):
        returnValue = ExternalGeneratorFilter.__new__(type(self))
        returnValue.__init__(self._prod.clone())
        return returnValue
    def insertInto(self, parameterSet, myname):
        newpset = parameterSet.newPSet()
        newpset.addString(True, "@module_label", self.moduleLabel_(myname))
        newpset.addString(True, "@module_type", self.type_())
        newpset.addString(True, "@module_edm_type", cms.EDFilter.__name__)
        newpset.addString(True, "@external_type", self._prod.type_())
        newpset.addString(False,"@python_config", self._prod.dumpPython())
        newpset.addBool(False,"_external_process_verbose_", self._external_process_verbose_.value())
        newpset.addUInt32(False,"_external_process_waitTime_", self._external_process_waitTime_.value())
        newpset.addVString(True, "_external_process_components_", self._external_process_components_.value())

        extraConfig =''
        for x in self._external_process_components_.value():
            extraConfig += "process."+x+"="+parameterSet.getTopPSet_(x).dumpPython()
            extraConfig += '\n'
        newpset.addString(False, "_external_process_extraConfig_", extraConfig)
        self._prod.insertContentsInto(newpset)
        parameterSet.addPSet(True, self.nameInProcessDesc_(myname), newpset)
    def dumpPython(self, options=cms.PrintOptions()):
        cms.specialImportRegistry.registerUse(self)
        result = "%s(" % self.__class__.__name__ # not including cms. since the deriving classes are not in cms "namespace"
        options.indent()
        result += "\n"+options.indentation() + self._prod.dumpPython(options)
        result +=options.indentation()+",\n"
        result += options.indentation() + "_external_process_waitTime_ = " + self._external_process_waitTime_.dumpPython(options) + ',\n'
        result += options.indentation() + "_external_process_verbose_ = " + self._external_process_verbose_.dumpPython(options) + ',\n'
        result += options.indentation() + "_external_process_components_ =" + self._external_process_components_.dumpPython(options) + ','
        options.unindent()
        result += "\n)\n"
        return result

cms.specialImportRegistry.registerSpecialImportForType(ExternalGeneratorFilter, "from GeneratorInterface.Core.ExternalGeneratorFilter import ExternalGeneratorFilter")
