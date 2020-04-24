from CalibTracker.SiStripCommon.shallowTree_test_template import *
process.TFileService.fileName = 'test_shallowTrackClustersProducer.root'

from RecoTracker.TrackProducer.TrackRefitter_cfi import TrackRefitter

process.load('CalibTracker.SiStripCommon.ShallowTrackClustersProducer_cfi')
process.load('RecoTracker.TrackProducer.TrackRefitters_cff')
process.tracksRefit = TrackRefitter.clone()
process.shallowTrackClusters.Tracks = 'tracksRefit'

process.testTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_shallowTrackClusters_*_*',
      )
   )
process.p = cms.Path(process.MeasurementTrackerEvent*process.tracksRefit*process.shallowTrackClusters*process.testTree)
