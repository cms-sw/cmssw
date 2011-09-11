import FWCore.ParameterSet.Config as cms

process = cms.Process("BeamMonitor")

#----------------------------
# Event Source
#-----------------------------
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('HLT_L1*',
                               'HLT_Jet*',
                               'HLT_*Cosmic*',
                               'HLT_HT*',
                               'HLT_MinBias_*',
                               'HLT_Physics*',
                               'HLT_ZeroBias_v2')
                              )

#--------------------------
# Filters
#--------------------------
# HLT Filter
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1


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
process.load("DQM.BeamMonitor.BeamMonitorBx_cff")
process.load("DQM.BeamMonitor.BeamMonitor_PixelLess_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")
# NB that resetEveryNLumi means the number of lumis to use for the *running
# average*
process.dqmBeamMonitor.resetEveryNLumi = 5
process.dqmBeamMonitor.resetPVEveryNLumi = 5
process.dqmBeamMonitorBx.fitEveryNLumi = 30
process.dqmBeamMonitorBx.resetEveryNLumi = 30


#TriggerName for selecting pv for DIP publication, NO wildcard needed here
#it will pick all triggers which has these strings in theri name
process.dqmBeamMonitor.jetTrigger  = cms.untracked.vstring("HLT_ZeroBias_v",
                                                           "HLT_Jet300_v",
                                                           "HLT_QuadJet60_v")

process.dqmBeamMonitor.hltResults = cms.InputTag("TriggerResults","","HLT")


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
process.load("RecoTracker.IterativeTracking.iterativeTk_cff")
#process.load("Configuration.GlobalRuns.reco_TLR_311X")

## Pixelless Tracking
process.load('RecoTracker/Configuration/RecoTrackerNotStandard_cff')
process.MeasurementTracker.pixelClusterProducer = cms.string("")

# Offline Beam Spot
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

## Offline PrimaryVertices
import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
process.offlinePrimaryVertices = RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()

#process.offlinePrimaryVertices.TrackLabel = cms.InputTag("ctfPixelLess")

process.dqmBeamMonitor.BeamFitter.TrackCollection = cms.untracked.InputTag('generalTracks')
process.dqmBeamMonitorBx.BeamFitter.TrackCollection = cms.untracked.InputTag('generalTracks')
process.offlinePrimaryVertices.TrackLabel = cms.InputTag("generalTracks")

## Not sure if these lines are needed Will remove later
#process.initialStepSeeds.ClusterCheckPSet.MaxNumberOfPixelClusters=2000
#process.newSeedFromPairs.ClusterCheckPSet.MaxNumberOfCosmicClusters=10000
#process.secTriplets.ClusterCheckPSet.MaxNumberOfPixelClusters=2000
#process.fifthSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 10000
#process.fourthPLSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters=10000


#---- replaces ----
process.initialStepSeeds.RegionFactoryPSet.ComponentName = 'GlobalRegionProducerFromBeamSpot' # was GlobalRegionProducer
#---- new parameters ----
process.initialStepSeeds.RegionFactoryPSet.RegionPSet.beamSpot = cms.InputTag("offlineBeamSpot")

#process.tracking = cms.Sequence(process.siPixelDigis
#                                *process.siStripDigis
#                                *process.trackerlocalreco
#                                *process.offlineBeamSpot
#                                *process.recopixelvertexing
#                                *process.ckftracks)

#process.tracking_pixelless = cms.Sequence(process.siPixelDigis
#                                           *process.siStripDigis
#                                           *process.trackerlocalreco
#                                           *process.offlineBeamSpot
#                                           *process.ctfTracksPixelLess)


#fast general track reco
process.iterTracking =cms.Sequence(process.InitialStep
                                  *process.LowPtTripletStep
                                  *process.PixelPairStep
                                  *process.DetachedTripletStep
                                  *process.MixedTripletStep
                                  *process.PixelLessStep
                                  *process.TobTecStep
                                  *process.generalTracks) 


process.tracking_FirstStep = cms.Sequence(process.siPixelDigis
                                         *process.siStripDigis
                                         *process.trackerlocalreco
                                         *process.offlineBeamSpot
                                         *process.recopixelvertexing
                                         *process.iterTracking)


#### END OF TRACKING RECONSTRUCTION ####





# Change Beam Monitor variables
if process.dqmSaver.producer.value() is "Playback":
  process.dqmBeamMonitor.BeamFitter.WriteAscii = False
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmdev/BeamMonitorDQM/BeamFitResults.txt'
else:
  process.dqmBeamMonitor.BeamFitter.WriteAscii = True
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults.txt'
#process.dqmBeamMonitor.BeamFitter.SaveFitResults = False
#process.dqmBeamMonitor.BeamFitter.OutputFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.root'
  process.dqmBeamMonitorBx.BeamFitter.WriteAscii = True
  process.dqmBeamMonitorBx.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults_Bx.txt'

#process.dqmBeamMonitor.BeamFitter.InputBeamWidth = 0.006
process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 25

## TKStatus
process.dqmTKStatus = cms.EDAnalyzer("TKStatus",
        BeamFitter = cms.PSet(
        DIPFileName = process.dqmBeamMonitor.BeamFitter.DIPFileName
        )
)



#--------------------------
# Path
#--------------------------


process.dqmcommon = cms.Sequence(process.dqmEnv
                                *process.dqmSaver)

process.monitor = cms.Sequence(process.dqmBeamMonitor)


#process.monitor_pixelless = cms.Sequence(process.dqmBeamMonitor_pixelless
#                                          *process.dqmEnvPixelLess)

process.p = cms.Path(process.scalersRawToDigi
                    *process.dqmTKStatus
                    *process.hltTriggerTypeFilter
                    *process.dqmcommon
                    *process.tracking_FirstStep
                    *process.offlinePrimaryVertices
                    *process.monitor)

