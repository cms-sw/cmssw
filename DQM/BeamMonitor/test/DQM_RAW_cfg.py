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

## Offline PrimaryVertices
import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
process.offlinePrimaryVertices = RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()

## DQM environment for pixelLess tracks
import DQMServices.Components.DQMEnvironment_cfi
process.dqmEnvPixelLess = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
process.dqmEnvPixelLess.subSystemFolder = 'BeamMonitor_PixelLess'
### conditions
process.GlobalTag.globaltag = 'GR09_R_34X_V2::All'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

 '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_InterestingEvents-Dec19thSkim_341_v1/0005/FC99B35E-ACED-DE11-B966-002618943879.root'
,'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_InterestingEvents-Dec19thSkim_341_v1/0005/CAB58842-AAED-DE11-AFDD-001A92971AD0.root'
,'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_InterestingEvents-Dec19thSkim_341_v1/0005/C0C8BB42-AAED-DE11-9F6C-00261894392C.root'

    )
)

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
process.p = cms.Path(process.RecoForDQM*process.offlinePrimaryVertices*process.BeamMonitorDQM*process.dqmSaver)

## Pixelless Tracking
#process.p = cms.Path(process.RecoForDQM_Pixelless*process.offlinePrimaryVertices*process.BeamMonitorDQM_Pixelless*process.dqmSaver)

## Both Tracking
#process.p = cms.Path(process.RecoForDQM_ALL*process.offlinePrimaryVertices*process.BeamMonitorDQM+process.BeamMonitorDQM_Pixelless+process.dqmSaver)

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

