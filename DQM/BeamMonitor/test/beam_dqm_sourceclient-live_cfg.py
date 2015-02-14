import FWCore.ParameterSet.Config as cms

process = cms.Process("BeamMonitor")

#----------------------------
# Event Source
#-----------------------------
process.load("DQM.Integration.test.inputsource_cfi")
#process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('HLT_MinBiasBSC','HLT_L1_BSC'))

#--------------------------
# Filters
#--------------------------
# HLT Filter
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1

# L1 Trigger Bit Selection (bit 40 and 41 for BSC trigger)
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('40 OR 41')

#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamMonitor'

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
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

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

## Offline PrimaryVertices
import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
process.offlinePrimaryVertices = RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()
## Input track for PrimaryVertex reconstruction, uncomment the following line to use pixelLess tracks
#process.offlinePrimaryVertices.TrackLabel = cms.InputTag("ctfPixelLess")
process.dqmBeamMonitor.BeamFitter.TrackCollection = cms.untracked.InputTag('firstStepTracksWithQuality')
process.offlinePrimaryVertices.TrackLabel = cms.InputTag("firstStepTracksWithQuality")

#### END OF TRACKING RECONSTRUCTION ####

# Change Beam Monitor variables
process.dqmBeamMonitor.BeamFitter.WriteAscii = True
process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
process.dqmBeamMonitor.BeamFitter.SaveFitResults = True
process.dqmBeamMonitor.BeamFitter.OutputFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.root'

#--------------------------
# Scheduling
#--------------------------
process.phystrigger = cms.Sequence(process.hltTriggerTypeFilter*process.gtDigis*process.hltLevel1GTSeed)
process.tracking = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks)
process.monitor = cms.Sequence(process.dqmBeamMonitor*process.dqmEnv)
process.tracking_pixelless = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.ctfTracksPixelLess)
process.monitor_pixelless = cms.Sequence(process.dqmBeamMonitor_pixelless*process.dqmEnvPixelLess)
process.tracking_FirstStep = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.recopixelvertexing*process.firstStep)

process.p = cms.Path(process.gtDigis*process.tracking_FirstStep*process.offlinePrimaryVertices*process.monitor*process.dqmSaver)
#process.p = cms.Path(process.gtDigis*process.tracking*process.offlinePrimaryVertices*process.monitor*process.dqmSaver)
#process.p = cms.Path(process.phystrigger*process.tracking*process.offlinePrimaryVertices*process.monitor*process.dqmSaver)
#process.p = cms.Path(process.phystrigger*process.tracking_pixelless*process.offlinePrimaryVertices*process.monitor_pixelless*process.dqmSaver)

