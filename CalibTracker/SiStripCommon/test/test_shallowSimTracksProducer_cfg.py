from CalibTracker.SiStripCommon.shallowTree_test_template import *
process.TFileService.fileName = 'test_shallowSimTracksProducer.root'

process.load('CalibTracker.SiStripCommon.ShallowSimTracksProducer_cfi')
process.load('SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi')
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")

add_rawRelVals(process)

process.testTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_shallowSimTracks_*_*',
      )
   )
process.p = cms.Path(
   process.simHitTPAssocProducer*
   process.trackAssociatorByHits*
   process.shallowSimTracks*
   process.testTree
   )
