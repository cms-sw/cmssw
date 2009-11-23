import FWCore.ParameterSet.Config as cms

process = cms.Process("Beam Monitor")

#----------------------------
# DQM Environment
#-----------------------------
process.load("DQM.Integration.test.environment_cfi")

#----------------------------
# BeamMonitor
#-----------------------------
process.load("DQM.BeamMonitor.BeamMonitor_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")

####  SETUP TRACKING RECONSTRUCTION ####

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.GeometryPilot2_cff")

#-----------------------------
# Magnetic Field
#-----------------------------
#process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')

#--------------------------
# Calibration
#--------------------------
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

#-----------------------
#  Reconstruction Modules
#-----------------------
## Collision Reconstruction
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

# offline beam spot
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

#### END OF TRACKING RECONSTRUCTION ####

#----------------------------
# Event Source
#-----------------------------
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(

    )
)

process.tracking = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks)

process.monitor = cms.Sequence(process.dqmBeamMonitor*process.dqmEnv*process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = 'srv-c2d05-18'
process.DQM.collectorPort = 9090
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.p = cms.Path(process.tracking*process.monitor)

#process.out = cms.OutputModule("PoolOutputModule",
#                               fileName = cms.untracked.string('test.root'),
#			       outputCommands = cms.untracked.vstring(
#				'drop *',
#				'keep *_offlineBeamSpot_*_*',
#				'keep *_ctfWithMaterialTracksP5_*_*',
#				'keep *_cosmictrackfinderP5_*_*'
#			       )
#                              )
#process.end = cms.EndPath(process.out)


