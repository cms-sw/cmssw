from CalibTracker.SiStripCommon.shallowTree_test_template import *
process.TFileService.fileName = 'test_shallowSimhitClustersProducer.root'

add_rawRelVals(process)

process.load('CalibTracker.SiStripCommon.ShallowSimhitClustersProducer_cfi')
process.testTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_shallowSimhitClusters_*_*',
      )
   )
process.p = cms.Path(process.shallowSimhitClusters*process.testTree)
