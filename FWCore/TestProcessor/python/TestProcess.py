import FWCore.ParameterSet.Config as cms

class TestProcess(cms.Process):
    def __init__(self,name="TEST",*modifiers):
        super(TestProcess,self).__init__(name,*modifiers)
        self.__dict__["_TestProcess__moduleToTest"] = None
    def moduleToTest(self,mod,task=cms.Task()):
        self.__dict__["_TestProcess__moduleToTest"] = mod
        if isinstance(mod,cms.EDFilter):
          self._test_path = cms.Path(mod,task)
        else:
          self._test_endpath = cms.EndPath(mod,task)
    def fillProcessDesc(self, processPSet):
        if self.__dict__["_TestProcess__moduleToTest"] is None:
            raise LogicError("moduleToTest was not called")
        for p in self.paths.iterkeys():
            if p != "_test_path":
                delattr(self,p)
        for p in self.endpaths.iterkeys():
            if p != "_test_endpath":
                delattr(self,p)
        if not hasattr(self,"options"):
            self.options = cms.untracked.PSet()
        cms.Process.fillProcessDesc(self,processPSet)
        processPSet.addString(True, "@moduleToTest",self.__dict__["_TestProcess__moduleToTest"].label_())
