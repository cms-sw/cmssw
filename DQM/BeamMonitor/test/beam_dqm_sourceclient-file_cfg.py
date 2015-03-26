import FWCore.ParameterSet.Config as cms

process = cms.Process("BeamMonitor")

#----------------------------
# Event Source
#-----------------------------
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(

'file:/lookarea_SM/Data.00130792.0001.Express.storageManager.00.0000.dat',
'file:/lookarea_SM/Data.00130792.0021.Express.storageManager.01.0000.dat',
'file:/lookarea_SM/Data.00130792.0041.Express.storageManager.02.0000.dat',
'file:/lookarea_SM/Data.00130792.0061.Express.storageManager.03.0000.dat',
'file:/lookarea_SM/Data.00130792.0081.Express.storageManager.04.0000.dat',
'file:/lookarea_SM/Data.00130792.0101.Express.storageManager.05.0000.dat'

    )
)
#process.NewEventStreamFileReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('HLT_MinBiasBSC','HLT_L1_BSC'))

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
#process.load("DQM.BeamMonitor.BeamMonitor_Cosmics_cff")
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

## Cosmic Track Reconstruction
process.load("RecoTracker.Configuration.RecoTrackerP5_cff")

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

#### END OF TRACKING RECONSTRUCTION ####

# Change Beam Monitor variables
process.dqmBeamMonitor.Debug = True
process.dqmBeamMonitor.BeamFitter.Debug = True
process.dqmBeamMonitor_pixelless.Debug = True
process.dqmBeamMonitor_pixelless.BeamFitter.Debug = True
process.dqmBeamMonitor.BeamFitter.WriteAscii = True
process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
#process.dqmBeamMonitor.BeamFitter.SaveFitResults = True
process.dqmBeamMonitor.BeamFitter.OutputFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.root'

#--------------------------
# Scheduling
#--------------------------
process.phystrigger = cms.Sequence(process.hltTriggerTypeFilter*process.gtDigis*process.hltLevel1GTSeed)
process.tracking = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks)
process.monitor = cms.Sequence(process.dqmBeamMonitor*process.dqmEnv)
process.tracking_pixelless = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.ctfTracksPixelLess)
process.monitor_pixelless = cms.Sequence(process.dqmBeamMonitor_pixelless*process.dqmEnvPixelLess)
## Cosmic just for testing DQM
process.tracking_cosmic = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.ctftracksP5)

## Cosmic
process.monitor_cosmic = cms.Sequence(process.dqmBeamMonitor*process.dqmEnv)
#process.offlinePrimaryVertices.TrackLabel = cms.InputTag("ctfWithMaterialTracksP5")

##FirstStep
process.tracking_FirstStep = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.recopixelvertexing*process.firstStep)
process.dqmBeamMonitor.BeamFitter.TrackCollection = cms.untracked.InputTag('firstStepTracksWithQuality')
process.offlinePrimaryVertices.TrackLabel = cms.InputTag("firstStepTracksWithQuality")

## Change FirstStep default values
# Step 0
process.newSeedFromTriplets.RegionFactoryPSet.RegionPSet.ptMin = 1.2 ## default : 0.8
process.newTrajectoryFilter.filterPset.minPt = 1.0 ## default : 0.6
# Step 1
process.newSeedFromPairs.RegionFactoryPSet.RegionPSet.ptMin = 1.2 ## default : 0.9
process.stepOneTrajectoryFilter.filterPset.minPt = 1.0 ## default : 0.5

process.p = cms.Path(process.gtDigis*process.tracking_FirstStep*process.offlinePrimaryVertices*process.monitor*process.dqmSaver)
#process.p = cms.Path(process.tracking_cosmic*process.offlinePrimaryVertices*process.monitor_cosmic*process.dqmSaver)
#process.p = cms.Path(process.tracking*process.offlinePrimaryVertices*process.monitor*process.dqmSaver)
#process.p = cms.Path(process.phystrigger*process.tracking*process.offlinePrimaryVertices*process.monitor*process.dqmSaver)
#process.p = cms.Path(process.phystrigger*process.tracking_pixelless*process.offlinePrimaryVertices*process.monitor_pixelless*process.dqmSaver)
# For test
process.dqmSaver.dirName = '.'
#process.p = cms.Path(process.tracking*process.offlinePrimaryVertices*process.monitor*process.dqmSaver)

## summary
#process.options = cms.untracked.PSet(
#    wantSummary = cms.untracked.bool(True)
#    )


