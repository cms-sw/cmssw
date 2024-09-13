import FWCore.ParameterSet.Config as cms

class TestSourceProcess(cms.Process):
    def __init__(self,name="TEST",*modifiers):
        super(TestSourceProcess,self).__init__(name,*modifiers)
    def fillProcessDesc(self, processPSet):
        if not hasattr(self,"options"):
            self.options = cms.untracked.PSet()
        cms.Process.fillProcessDesc(self,processPSet)
