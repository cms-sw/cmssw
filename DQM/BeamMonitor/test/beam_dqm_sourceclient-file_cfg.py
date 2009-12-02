import FWCore.ParameterSet.Config as cms

process = cms.Process("BeamMonitor")

#----------------------------
# DQM Environment
#-----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

import DQMServices.Components.DQMEnvironment_cfi
process.dqmEnvPixelLess = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
process.dqmEnvPixelLess.subSystemFolder = 'BeamMonitor_PixelLess'

#----------------------------
# BeamMonitor
#-----------------------------
process.load("DQM.BeamMonitor.BeamMonitor_cff")
process.load("DQM.BeamMonitor.BeamMonitor_PixelLess_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")

####  SETUP TRACKING RECONSTRUCTION ####

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
process.load("Configuration.StandardSequences.Geometry_cff")

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

## Pixelless Tracking
process.load('RecoTracker/Configuration/RecoTrackerNotStandard_cff')
process.MeasurementTracker.pixelClusterProducer = cms.string("")

# Offline Beam Spot
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

process.NewEventStreamFileReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('HLT_MinBia*','HLT_L1*'))

#--------------------------
# Filters
#--------------------------
# HLT Filter
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1

#--------------------------
# Scheduling
#--------------------------
process.tracking = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks)
process.monitor = cms.Sequence(process.dqmBeamMonitor*process.dqmEnv)
process.tracking_pixelless = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.ctfTracksPixelLess)
process.monitor_pixelless = cms.Sequence(process.dqmBeamMonitor_pixelless*process.dqmEnvPixelLess)

#process.p = cms.Path(process.hltTriggerTypeFilter*process.tracking*process.monitor)
#process.p = cms.Path(process.hltTriggerTypeFilter*process.tracking_pixelless*process.monitor_pixelless)
process.p = cms.Path(process.hltTriggerTypeFilter*process.tracking*process.monitor+process.tracking_pixelless*process.monitor_pixelless+process.dqmSaver)


