from CalibTracker.SiStripCommon.shallowTree_test_template import *
process.TFileService.fileName = 'test_shallowRechitClustersProducer.root'
process.load('RecoTracker.TrackProducer.TrackRefitters_cff')
process.load('CalibTracker.SiStripCommon.ShallowRechitClustersProducer_cfi')

process.testTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_shallowRechitClusters_*_*',
      )
   )
process.p = cms.Path(
   process.siStripMatchedRecHits*
   process.shallowRechitClusters*
   process.testTree
)
