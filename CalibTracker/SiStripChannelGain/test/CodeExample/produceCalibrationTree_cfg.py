import FWCore.ParameterSet.Config as cms

process = cms.Process('CALIB')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'
#process.GlobalTag.globaltag = "GR09_31X_V6P::All"

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet( threshold = cms.untracked.string('ERROR')   ),
    destinations = cms.untracked.vstring('cout')
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.add_( cms.Service( "TFileService",
                           fileName = cms.string( 'XXX_OUTPUT_XXX' ),
                           closeFileFast = cms.untracked.bool(True)
) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source (
    "PoolSource",
     fileNames = cms.untracked.vstring(
        XXX_INPUT_XXX
     ),
    secondaryFileNames = cms.untracked.vstring()
)

process.load('CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi')
#process.load('CalibTracker.SiStripCommon.ShallowClustersProducer_cfi')
#process.load('CalibTracker.SiStripCommon.ShallowTrackClustersProducer_cfi')
#process.load('CalibTracker.SiStripCommon.ShallowRechitClustersProducer_cfi')
process.load('CalibTracker.SiStripCommon.ShallowTracksProducer_cfi')
process.load('CalibTracker.SiStripCommon.ShallowGainCalibration_cfi')


process.shallowTree = cms.EDAnalyzer("ShallowTree",
     outputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_shallowEventRun_*_*',
    'keep *_shallowTracks_trackchi2ndof_*',
    'keep *_shallowTracks_trackmomentum_*',
    'keep *_shallowTracks_trackpt_*',
    'keep *_shallowTracks_tracketa_*',
    'keep *_shallowTracks_trackphi_*',
    'keep *_shallowTracks_trackhitsvalid_*',
    'keep *_shallowGainCalibration_*_*'
    )
)

process.load('CalibTracker.Configuration.Filter_Refit_cff')
#process.load('CalibTracker.SiStripCommon.theBigNtuple_cfi')
#process.shallowTrackClusters.Tracks   = "CalibrationTracksRefit"
#process.shallowTrackClusters.Clusters = 'CalibrationTracks'
#process.shallowClusters.Clusters      = 'CalibrationTracks'
process.shallowTracks.Tracks          = "CalibrationTracksRefit"
process.shallowGainCalibration.Tracks = 'CalibrationTracksRefit'

process.p = cms.Path(process.trackFilterRefit + process.shallowEventRun + process.shallowTracks  + process.shallowGainCalibration + process.shallowTree)

