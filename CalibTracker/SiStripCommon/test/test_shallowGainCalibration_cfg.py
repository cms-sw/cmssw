from CalibTracker.SiStripCommon.shallowTree_test_template import *
process.TFileService.fileName = 'test_shallowGainCalibration.root'

process.load('CalibTracker.SiStripCommon.ShallowGainCalibration_cfi')
from RecoTracker.TrackProducer.TrackRefitter_cfi import TrackRefitter

process.load('RecoTracker.TrackProducer.TrackRefitters_cff')
process.tracksRefit = TrackRefitter.clone()
process.shallowGainCalibration.Tracks = 'tracksRefit'

process.testTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_GainCalibration_*_*',
      )
   )
process.p = cms.Path(process.MeasurementTrackerEvent*process.tracksRefit*process.shallowGainCalibration*process.testTree)
