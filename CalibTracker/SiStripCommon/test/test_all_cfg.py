#! /bin/env cmsRun

from CalibTracker.SiStripCommon.shallowTree_test_template import *
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(3) )
process.TFileService.fileName = 'test_all.root'

process.load('CalibTracker.SiStripCommon.ShallowGainCalibration_cfi')
process.load('CalibTracker.SiStripCommon.ShallowDigisProducer_cfi')
process.load('CalibTracker.SiStripCommon.ShallowTracksProducer_cfi')
process.load('CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi')
process.load('CalibTracker.SiStripCommon.ShallowTrackClustersProducer_cfi')
process.load('CalibTracker.SiStripCommon.ShallowSimTracksProducer_cfi')
process.load('CalibTracker.SiStripCommon.ShallowSimhitClustersProducer_cfi')
process.load('CalibTracker.SiStripCommon.ShallowClustersProducer_cfi')
process.load('CalibTracker.SiStripCommon.ShallowRechitClustersProducer_cfi')

process.load('RecoTracker.TrackProducer.TrackRefitters_cff')
process.load('SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi')
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
#add_rawRelVals(process)

process.tracksRefit = process.TrackRefitter.clone()
process.shallowGainCalibration.Tracks = 'tracksRefit'
process.shallowTrackClusters.Tracks = 'tracksRefit'

process.testTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_shallowGainCalibration_*_*',
      'keep *_shallowDigis_*_*',
      'keep *_shallowTracks_*_*',
      'keep *_shallowEventRun_*_*',
      'keep *_shallowTrackClusters_*_*',
      'keep *_shallowSimTracks_*_*',
      'keep *_shallowSimhitClusters_*_*',
      'keep *_shallowClusters_*_*',
      'keep *_shallowRechitClusters_*_*',
      )
   )

process.p = cms.Path(
   process.MeasurementTrackerEvent*
   process.tracksRefit*
   #process.simHitTPAssocProducer*
   #process.trackAssociatorByHits*
   process.siStripMatchedRecHits*
   # Shallow stuff
   process.shallowEventRun*
   #process.shallowSimTracks*
   process.shallowTrackClusters*
   process.shallowGainCalibration*
   process.shallowDigis*
   process.shallowTracks*
   #process.shallowSimhitClusters*
   process.shallowClusters*
   process.shallowRechitClusters*
   #tree dumping
   process.testTree
)
