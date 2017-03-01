import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
###DQM Envirionment####################
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.BeamMonitor.BeamMonitor_cff")
process.load("DQM.BeamMonitor.BeamMonitorBx_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")

process.load("DQMServices.Components.DQMEnvironment_cfi")

#######----Event to Analyze----########
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

####---Massage logger Services----####
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['dqmBeamMonitor']
process.MessageLogger.categories = ['BeamMonitor']
process.MessageLogger.cerr.threshold = "INFO"


#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamMonitor'

import DQMServices.Components.DQMEnvironment_cfi
process.dqmEnvPixelLess = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
process.dqmEnvPixelLess.subSystemFolder = 'BeamMonitor_PixelLess'


process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('HLT_L1*','HLT_TrackerCosmics','HLT_Jet*'))


process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
   'file:/tmp/schauhan/BACB7413-12FE-DF11-B0FB-003048678A6C.root'

 ),
  skipBadFiles = cms.untracked.bool(True),  
)


#--------------------------
# Filters
#--------------------------
# HLT Filter
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1

#----------------------------
# BeamMonitor
#-----------------------------
process.dqmBeamMonitor.OnlineMode        = True
process.dqmBeamMonitor.resetEveryNLumi   = 5
process.dqmBeamMonitor.resetPVEveryNLumi = 5
#process.dqmBeamMonitor.BeamFitter.MinimumInputTracks = 150

# Set Lower if the bunches are low
process.dqmBeamMonitorBx.fitEveryNLumi   = 30
process.dqmBeamMonitorBx.resetEveryNLumi = 30

#TriggerName for selecting pv for DIP
process.dqmBeamMonitor.jetTrigger  = cms.untracked.vstring("HLT_L1",
                                                           "HLT_Act")

process.dqmBeamMonitor.hltResults = cms.InputTag("TriggerResults","","HLT")
#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-----------------------------
# Magnetic Field
#-----------------------------
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')



#-----------------------
#  Reconstruction Modules
#-----------------------
## Collision Reconstruction
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoTracker.IterativeTracking.iterativeTk_cff")

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

process.dqmBeamMonitor.BeamFitter.TrackCollection = cms.untracked.InputTag('generalTracks')
process.dqmBeamMonitorBx.BeamFitter.TrackCollection = cms.untracked.InputTag('generalTracks')
process.offlinePrimaryVertices.TrackLabel = cms.InputTag("generalTracks")



#---- replaces ----
process.initialStepSeeds.RegionFactoryPSet.ComponentName = 'GlobalRegionProducerFromBeamSpot' # was GlobalRegionProducer
#---- new parameters ----
process.initialStepSeeds.RegionFactoryPSet.RegionPSet.beamSpot = cms.InputTag("offlineBeamSpot")



#process.tracking = cms.Sequence(process.siPixelDigis
#                                 *process.siStripDigis
#                                 *process.trackerlocalreco
#                                 *process.offlineBeamSpot
#                                 *process.recopixelvertexing
#                                 *process.ckftracks)

##For Pixel Less tracking
#process.tracking_pixelless = cms.Sequence(process.siPixelDigis
#                                          *process.siStripDigis
#                                          *process.trackerlocalreco
#                                          *process.offlineBeamSpot
#                                          *process.ctfTracksPixelLess)



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
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = 'BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = 'DIP_BeamFitResults.txt'
else:
  process.dqmBeamMonitor.BeamFitter.WriteAscii = True
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = 'BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = 'DIP_BeamFitResults.txt'
#process.dqmBeamMonitor.BeamFitter.SaveFitResults = False
#process.dqmBeamMonitor.BeamFitter.OutputFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.root'
  process.dqmBeamMonitorBx.BeamFitter.WriteAscii = True
  process.dqmBeamMonitorBx.BeamFitter.AsciiFileName = 'BeamFitResults_Bx.txt'

# Lower for HI
process.dqmBeamMonitor.PVFitter.minNrVerticesForFit   = 20
process.dqmBeamMonitorBx.PVFitter.minNrVerticesForFit = 20


## TKStatus
process.dqmTKStatus = cms.EDAnalyzer("TKStatus",
        BeamFitter = cms.PSet(
        DIPFileName = process.dqmBeamMonitor.BeamFitter.DIPFileName
        )
)



#--------------------------
#process.phystrigger = cms.Sequence(process.hltTriggerTypeFilter*process.gtDigis*process.hltLevel1GTSeed)
process.dqmcommon = cms.Sequence(process.dqmEnv
                                 *process.dqmSaver)

process.monitor = cms.Sequence(process.dqmBeamMonitor
                               )

#process.monitor_pixelless = cms.Sequence(process.dqmBeamMonitor_pixelless
#                                          *process.dqmEnvPixelLess)


# Setup DQM store parameters.
process.DQMStore.verbose = 1
#process.DQM.collectorHost = 'cmslpc17.fnal.gov'  ## at cmslpc
process.DQM.collectorHost   = 'lxplus438.cern.ch'
process.DQM.collectorPort   = 9190
process.dqmSaver.dirName    = '.'
process.dqmSaver.producer   = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun     = 1
process.dqmSaver.saveAtJobEnd  = True

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_44_V4::All'


process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.pp = cms.Path( process.scalersRawToDigi
                      *process.dqmTKStatus
                      *process.hltTriggerTypeFilter
                      *process.dqmcommon
                      *process.tracking_FirstStep
                      *process.offlinePrimaryVertices
                      *process.monitor
                     )

 
#print process.dumpPython()
