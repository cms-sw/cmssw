import FWCore.ParameterSet.Config as cms

class TestProcess(cms.Process):
    def __init__(self,name="TEST",*modifiers):
        super(TestProcess,self).__init__(name,*modifiers)
        self.__dict__["_TestProcess__moduleToTest"] = None
    def moduleToTest(self,mod):
        self.__dict__["_TestProcess__moduleToTest"] = mod
        self._test_endpath = cms.EndPath(mod)
    def fillProcessDesc(self, processPSet):
        if self.__dict__["_TestProcess__moduleToTest"] is None:
            raise LogicError("moduleToTest was not called")
        for p in self.paths.iterkeys():
            delattr(self,p)
        for p in self.endpaths.iterkeys():
            if p != "_test_endpath":
                delattr(self,p)
        if not hasattr(self,"options"):
            self.options = cms.untracked.PSet()
        cms.Process.fillProcessDesc(self,processPSet)
        processPSet.addString(True, "@moduleToTest",self.__dict__["_TestProcess__moduleToTest"].label_())
