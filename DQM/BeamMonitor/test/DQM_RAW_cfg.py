import FWCore.ParameterSet.Config as cms

process = cms.Process("BeamMonitor")
## DQM common
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

## General common
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")

## Pixel less tracking
process.load('RecoTracker/Configuration/RecoTrackerNotStandard_cff')
process.MeasurementTracker.pixelClusterProducer = cms.string("")

## BeamMonitor modules
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")
process.load("DQM.BeamMonitor.BeamMonitor_cff")
process.dqmBeamMonitor.Debug = True
process.dqmBeamMonitor.BeamFitter.Debug = True
process.load("DQM.BeamMonitor.BeamMonitor_PixelLess_cff")
process.dqmBeamMonitor_pixelless.Debug = True
process.dqmBeamMonitor_pixelless.BeamFitter.Debug = True

process.dqmBeamMonitor.BeamFitter.WriteAscii = True
process.dqmBeamMonitor.BeamFitter.AsciiFileName = 'BeamFitResults.txt'
process.dqmBeamMonitor.BeamFitter.SaveFitResults = True
process.dqmBeamMonitor.BeamFitter.OutputFileName = 'BeamFitResults.root'

## Offline PrimaryVertices
import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
process.offlinePrimaryVertices = RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()

## DQM environment for pixelLess tracks
import DQMServices.Components.DQMEnvironment_cfi
process.dqmEnvPixelLess = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
process.dqmEnvPixelLess.subSystemFolder = 'BeamMonitor_PixelLess'
### conditions
process.GlobalTag.globaltag = 'GR09_R_34X_V2::All'

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


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/AC4043E8-A5ED-DE11-91FB-001A92971B32.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/AC4043E8-A5ED-DE11-91FB-001A92971B32.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/F8FD9DD9-A5ED-DE11-AF2C-0030486791AA.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/BCD59B61-ACED-DE11-A010-003048D3FC94.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/461F7E44-AAED-DE11-B5A4-002618943932.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/D2F80F5F-ACED-DE11-B3B6-0018F3D09630.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/BCD59B61-ACED-DE11-A010-003048D3FC94.root'

    )
)
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('124024:2-124024:83')

process.phystrigger = cms.Sequence(process.hltTriggerTypeFilter*process.gtDigis*process.hltLevel1GTSeed)
process.pretracking_step = cms.Sequence(process.siPixelDigis*
                                        process.siStripDigis*
                                        process.trackerlocalreco*
                                        process.offlineBeamSpot
                                       )

process.RecoForDQM = cms.Sequence(process.pretracking_step*process.recopixelvertexing*process.ckftracks)
process.RecoForDQM_Pixelless = cms.Sequence(process.pretracking_step+process.ctfTracksPixelLess)
process.RecoForDQM_ALL = cms.Sequence(process.pretracking_step+process.ckftracks+process.ctfTracksPixelLess)
process.BeamMonitorDQM = cms.Sequence(process.dqmBeamMonitor+process.dqmEnv)
process.BeamMonitorDQM_Pixelless = cms.Sequence(process.dqmBeamMonitor_pixelless+process.dqmEnvPixelLess)

## Input track for PrimaryVertex reconstruction, comment out the following line to use default generalTracks
#process.offlinePrimaryVertices.TrackLabel = cms.InputTag("ctfPixelLess")

## Normal Tracking
process.p = cms.Path(process.phystrigger*process.RecoForDQM*process.offlinePrimaryVertices*process.BeamMonitorDQM*process.dqmSaver)
#process.p = cms.Path(process.RecoForDQM*process.offlinePrimaryVertices*process.BeamMonitorDQM*process.dqmSaver)

## Pixelless Tracking
#process.p = cms.Path(process.phystrigger*process.RecoForDQM_Pixelless*process.offlinePrimaryVertices*process.BeamMonitorDQM_Pixelless*process.dqmSaver)

## Both Tracking
#process.p = cms.Path(process.phystrigger*process.RecoForDQM_ALL*process.offlinePrimaryVertices*process.BeamMonitorDQM+process.BeamMonitorDQM_Pixelless+process.dqmSaver)

## DQM settings
process.DQMStore.verbose = 0
process.DQM.collectorHost = 'cmslpc14.fnal.gov'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

## summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

