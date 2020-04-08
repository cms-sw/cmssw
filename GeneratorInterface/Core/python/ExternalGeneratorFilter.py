import FWCore.ParameterSet.Config as cms

class ExternalGeneratorFilter(cms.EDFilter):
    def __init__(self, prod):
        self.__dict__['_prod'] = prod
        super(cms.EDFilter,self).__init__('ExternalGeneratorFilter')
    def __setattr__(self, name, value):
        setattr(self._prod, name, value)
    def __getattr__(self, name):
        if name =='_prod':
            return self.__dict__['_prod']
        return getattr(self._prod, name)
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
        self._prod.insertContentsInto(newpset)
        parameterSet.addPSet(True, self.nameInProcessDesc_(myname), newpset)
