from CalibTracker.SiStripCommon.shallowTree_test_template import *
process.TFileService.fileName = 'test_shallowClustersProducer.root'

process.load('CalibTracker.SiStripCommon.ShallowClustersProducer_cfi')
process.testTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_shallowClusters_*_*',
      )
   )
process.p = cms.Path(process.shallowClusters*process.testTree)
