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
      'keep *_shallowGainCalibration_*_*',
      )
   )

process.out = cms.OutputModule(
   "PoolOutputModule",
   fileName = cms.untracked.string('test_shallowGainCalibration_edm.root'),
   dropMetaData = cms.untracked.string("DROPPED"),
   outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_shallowGainCalibration_*_*'
      ),
)

process.p = cms.Path(process.MeasurementTrackerEvent*process.tracksRefit*process.shallowGainCalibration*process.testTree)
process.end = cms.EndPath(process.out)
process.s = cms.Schedule(process.p, process.end)
