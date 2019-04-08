import libFWCorePythonFramework as _pf
import libFWCorePythonParameterSet as _pp

class CmsRun(object):
  def __init__(self,process):
    """Uses a cms.Process to setup an edm::EventProcessor
    """
    procDesc = _pp.ProcessDesc()
    process.fillProcessDesc(procDesc.pset())
    self._cppProcessor = _pf.PythonEventProcessor(procDesc)

  def run(self):
    """Process all the events
    """
    self._cppProcessor.run()

  def totalEvents(self):
    return self._cppProcessor.totalEvents()
  def totalEventsPassed(self):
    return self._cppProcessor.totalEventsPassed()    
  def totalEventsFailed(self):
    return self._cppProcessor.totalEventsFailed()

if __name__ == "__main__":
  
  import unittest
  class testCmsRun(unittest.TestCase):
    def testFiltering(self):
      import FWCore.ParameterSet.Config as cms
      process = cms.Process("Test")
      process.source = cms.Source("EmptySource")
      nEvents=10
      process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(nEvents))
      process.scale = cms.EDFilter("Prescaler",prescaleFactor = cms.int32(1), prescaleOffset = cms.int32(0))
      process.p = cms.Path(process.scale)
      filterResults = ((10,0),(5,5),(3,7))
      for x in [1,2,3]:
        process.scale.prescaleFactor = x
        e = CmsRun(process)
        e.run()
        self.assertEqual(e.totalEvents(),nEvents)
        self.assertEqual(e.totalEventsPassed(),filterResults[x-1][0])
        self.assertEqual(e.totalEventsFailed(),filterResults[x-1][1])
        del e

  unittest.main()
