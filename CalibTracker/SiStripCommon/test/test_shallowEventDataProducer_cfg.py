from CalibTracker.SiStripCommon.shallowTree_test_template import *
process.TFileService.fileName = 'test_shallowEventDataProducer.root'

process.load('CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi')
process.testTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_shallowEventRun_*_*',
      )
   )
process.p = cms.Path(process.shallowEventRun*process.testTree)
