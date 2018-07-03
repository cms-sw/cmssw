import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process("PixelTrackTimingDQMLive", eras.Run2_2017)

live=True # set to False for Offline testing
offlineTesting=not live

#----------------------------
# Event Source
#-----------------------------

# for live online DQM in P5
if (live):
    process.load("DQM.Integration.config.inputsource_cfi")
# for testing in lxplus
elif(offlineTesting):
    process.load("DQM.Integration.config.fileinputsource_cfi")

#print process.runType 

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQM.Integration.config.environment_cfi")

process.dqmEnv.subSystemFolder    = "TrackTimingPixelPhase1"
process.dqmSaver.tag = "TrackTimingPixelPhase1"
process.dqmSaver.backupLumiCount = 30

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.dqmEnvTr = DQMEDAnalyzer('DQMEventInfo',
                 subSystemFolder = cms.untracked.string('TrackTimingPixelPhase1'),
                 eventRateWindow = cms.untracked.double(0.5),
                 eventInfoFolder = cms.untracked.string('EventInfo')
)

#-----------------------------
# Magnetic Field
#-----------------------------

process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#--------------------------
# Calibration
#--------------------------
# Condition for P5 cluster

if (live):
    process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
elif(offlineTesting):
    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
    from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
    process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')
                                                                                           
#-----------------------
#  Reconstruction Modules
#-----------------------

# Redirecting the SiPixelPhase1 output plots to under SiStrip
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *
DefaultHisto.enabled = True
DefaultHisto.topFolderName = "TrackTimingPixelPhase1" 
DefaultHistoDigiCluster.topFolderName=cms.string( "TrackTimingPixelPhase1/Phase1_MechanicalView")
DefaultHistoReadout.topFolderName=cms.string( "TrackTimingPixelPhase1/FED/Readout")
DefaultHistoTrack.topFolderName=cms.string( "TrackTimingPixelPhase1/Phase1_Track")

# PixelPhase1 Real data raw to digi
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.IncludeErrors = True

# PixelPhase1 Local Reconstruction
process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

# SiPixelTrack cosmics/pp settings 
process.load("DQM.SiPixelPhase1Config.SiPixelPhase1OnlineDQM_Timing_cff")

## Collision Reconstruction
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

## Cosmic Track Reconstruction
if (process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1):
    process.load("RecoTracker.Configuration.RecoTrackerP5_cff")
    process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
else:
    process.load("Configuration.StandardSequences.Reconstruction_cff")

import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

#--------------------------
# Service
#--------------------------
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Producers
#--------------------------

# Event History Producer
process.load("DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi")

# APV Phase Producer
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi")

#--------------------------
# Filters
#--------------------------

# HLT Filter
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1)
)

# L1 Trigger Bit Selection (bit 40 and 41 for BSC trigger)
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('NOT (36 OR 37 OR 38 OR 39)')

# HLT trigger selection (HLT_ZeroBias)
process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
process.hltHighLevel.HLTPaths = cms.vstring( 'HLT_ZeroBias_*' , 'HLT_ZeroBias1_*' , 'HLT_PAZeroBias_*' , 'HLT_PAZeroBias1_*', 'HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_*','HLT*SingleMu*')
process.hltHighLevel.andOr = cms.bool(True)
process.hltHighLevel.throw =  cms.bool(False)

#--------------------------
# Scheduling
#--------------------------

process.DQMCommon                = cms.Sequence(process.dqmEnv*process.dqmEnvTr*process.dqmSaver)
process.RecoForDQM_LocalReco     = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.gtDigis*process.trackerlocalreco)#*process.gtEvmDigis)

#--------------------------
# Global Plot Switches
#--------------------------

### COSMIC RUN SETTING
if (process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1):
    # Reference run for cosmic
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/sistrip_reference_cosmic.root'
        
    # Reco for cosmic data
    process.load('RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cfi')
    process.simpleCosmicBONSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 450
    process.combinatorialcosmicseedfinderP5.MaxNumberOfCosmicClusters = 450

    process.RecoForDQM_TrkReco_cosmic = cms.Sequence(process.offlineBeamSpot*process.MeasurementTrackerEvent*process.tracksP5)
    
    process.p = cms.Path(
                         ##### TRIGGER SELECTION #####
                         process.hltHighLevel*
                         process.scalersRawToDigi*
                         process.APVPhases*
                         process.consecutiveHEs*
                         process.hltTriggerTypeFilter*
                         process.RecoForDQM_LocalReco*
                         process.DQMCommon*
                         process.RecoForDQM_TrkReco_cosmic*
                         process.siPixelPhase1OnlineDQM_source_cosmics*
                         process.siPixelPhase1OnlineDQM_harvesting
                         )
   
### pp COLLISION SETTING

if (process.runType.getRunType() == process.runType.pp_run or process.runType.getRunType() == process.runType.pp_run_stage1):

    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/sistrip_reference_pp.root'
    # Source and Client config for pp collisions
    
    # Reco for pp collisions
    process.load('RecoTracker.IterativeTracking.InitialStepPreSplitting_cff')
    process.InitialStepPreSplitting.remove(process.initialStepTrackRefsForJetsPreSplitting)
    process.InitialStepPreSplitting.remove(process.caloTowerForTrkPreSplitting)
    process.InitialStepPreSplitting.remove(process.ak4CaloJetsForTrkPreSplitting)
    process.InitialStepPreSplitting.remove(process.jetsForCoreTrackingPreSplitting)
    process.InitialStepPreSplitting.remove(process.siPixelClusters)
    process.InitialStepPreSplitting.remove(process.siPixelRecHits)
    process.InitialStepPreSplitting.remove(process.MeasurementTrackerEvent)
    process.InitialStepPreSplitting.remove(process.siPixelClusterShapeCache)

    # Redefinition of siPixelClusters: has to be after RecoTracker.IterativeTracking.InitialStepPreSplitting_cff
    process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

    from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
    process.PixelLayerTriplets.BPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
    process.PixelLayerTriplets.FPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
    from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
    process.pixelTracksHitTriplets.SeedComparitorPSet.clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting'

    process.RecoForDQM_TrkReco = cms.Sequence(process.offlineBeamSpot*process.MeasurementTrackerEventPreSplitting*process.siPixelClusterShapeCachePreSplitting*process.recopixelvertexing*process.InitialStepPreSplitting)
    
    process.Reco = cms.Sequence(process.siPixelDigis*process.siPixelClusters)

    process.p = cms.Path(
        ##### TRIGGER SELECTION #####
        process.hltHighLevel*
        process.scalersRawToDigi*
        process.APVPhases*
        process.consecutiveHEs*
        process.hltTriggerTypeFilter*
        process.RecoForDQM_LocalReco*
        process.siPixelClusters*
        process.DQMCommon*
        process.RecoForDQM_TrkReco*
        process.siPixelPhase1OnlineDQM_source_pprun*
        process.siPixelPhase1OnlineDQM_harvesting
        )

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

