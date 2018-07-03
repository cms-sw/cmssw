from CalibTracker.SiStripCommon.shallowTree_test_template import *
process.TFileService.fileName = 'test_shallowDigisProducer.root'

process.load('CalibTracker.SiStripCommon.ShallowDigisProducer_cfi')
process.testTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_shallowDigis_*_*',
      )
   )
process.p = cms.Path(process.shallowDigis*process.testTree)
