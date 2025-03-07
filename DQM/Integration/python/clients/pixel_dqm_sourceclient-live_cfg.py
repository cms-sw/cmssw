import FWCore.ParameterSet.Config as cms

import sys
if 'runkey=hi_run' in sys.argv:
  from Configuration.Eras.Era_Run3_pp_on_PbPb_approxSiStripClusters_cff import Run3_pp_on_PbPb_approxSiStripClusters
  process = cms.Process("PIXELDQMLIVE", Run3_pp_on_PbPb_approxSiStripClusters)
else:
  from Configuration.Eras.Era_Run3_cff import Run3
  process = cms.Process("PIXELDQMLIVE", Run3)

live=True
unitTest = False

if 'unitTest=True' in sys.argv:
    live=False
    unitTest=True

#set to false for lxplus offline testing
#live=False
offlineTesting=not live

TAG ="PixelPhase1" 

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = cms.untracked.vstring('siPixelDigis',
                                                           'siStripClusters', 
                                                           'SiPixelRawDataErrorSource', 
                                                           'SiPixelDigiSource')
process.MessageLogger.cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR'))

#----------------------------
# Event Source
#-----------------------------
# for live online DQM in P5

if (unitTest):
    process.load("DQM.Integration.config.unittestinputsource_cfi")
    from DQM.Integration.config.unittestinputsource_cfi import options

elif (live):
    process.load("DQM.Integration.config.inputsource_cfi")
    from DQM.Integration.config.inputsource_cfi import options

# for testing in lxplus
elif(offlineTesting):
    process.load("DQM.Integration.config.fileinputsource_cfi")
    from DQM.Integration.config.fileinputsource_cfi import options

#-----------------------------
# DQM Environment
#-----------------------------

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQM.Integration.config.environment_cfi")

#----------------------------
# DQM Live Environment
#-----------------------------

process.dqmEnv.subSystemFolder = TAG
process.dqmSaver.tag = TAG
process.dqmSaver.runNumber = options.runNumber
# process.dqmSaverPB.tag = TAG
# process.dqmSaverPB.runNumber = options.runNumber


#-----------------------------
# Magnetic Field
#-----------------------------

process.load('Configuration.StandardSequences.MagneticField_cff')

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# GLOBALTAG
#-------------------------------------------------
# Condition for P5 cluster

if (live):
    process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# Condition for lxplus: change and possibly customise the GT
elif(offlineTesting):
    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
    from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
    process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

#-----------------------
#  Reconstruction Modules
#-----------------------

# Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi")
process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_RealData_cfi")

process.load("DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi")
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi")

# PixelPhase1 Real data raw to digi
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.IncludeErrors = True

if (process.runType.getRunType() == process.runType.hi_run):
  rawDataRepackerLabel = 'rawDataRepacker'
  #--------------------------------
  # Heavy Ion Configuration Changes
  #--------------------------------
  process.siPixelDigis.InputLabel = rawDataRepackerLabel
  process.siStripDigis.ProductLabel   = rawDataRepackerLabel
  process.scalersRawToDigi.scalersInputTag = rawDataRepackerLabel
  process.tcdsDigis.InputLabel = rawDataRepackerLabel
else :
  rawDataCollectorLabel = 'rawDataCollector'
  process.siPixelDigis.InputLabel = rawDataCollectorLabel
  process.siStripDigis.ProductLabel = rawDataCollectorLabel

## Collision Reconstruction
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

## Cosmic Track Reconstruction
if (process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1):
    process.load("RecoTracker.Configuration.RecoTrackerP5_cff")
    process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
    process.load("RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi")
else:
    process.load("Configuration.StandardSequences.Reconstruction_cff")

import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

process.load("DQM.SiPixelPhase1Config.SiPixelPhase1OnlineDQM_cff")

process.PerModule.enabled=True
process.PerReadout.enabled=True
process.OverlayCurvesForTiming.enabled=True
process.IsOffline.enabled=False

#--------------------------
# Service
#--------------------------
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Filters
#--------------------------

# HLT Filter
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1)
)

process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
if (process.runType.getRunType() == process.runType.hi_run):
    process.hltHighLevel.HLTPaths = ['HLT_ZeroBias_*' , 'HLT_HIZeroBias_*' , 'HLT_ZeroBias1_*' , 'HLT_PAZeroBias_*' , 'HLT_PAZeroBias1_*', 'HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_*','HLT*SingleMu*' , 'HLT_HICentralityVeto*' , 'HLT_HIMinimumBias*','HLT_HIPhysics*']
else:
    process.hltHighLevel.HLTPaths = ['HLT_ZeroBias_*' , 'HLT_ZeroBias1_*' , 'HLT_PAZeroBias_*' , 'HLT_PAZeroBias1_*', 'HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_*','HLT*SingleMu*']
process.hltHighLevel.andOr = True
process.hltHighLevel.throw =  False

#--------------------------
# Scheduling
#--------------------------

process.DQMmodules = cms.Sequence(process.dqmEnv* process.dqmSaver)#*process.dqmSaverPB)

process.RecoForDQM_LocalReco = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.gtDigis*process.trackerlocalreco)

### COSMIC RUN SETTING
if (process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1):
        
    # Reco for cosmic data
    process.load('RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cfi')
    process.simpleCosmicBONSeeds.ClusterCheckPSet.MaxNumberOfStripClusters = 450
    process.combinatorialcosmicseedfinderP5.MaxNumberOfStripClusters = 450

    

    process.RecoForDQM_TrkReco_cosmic = cms.Sequence(process.offlineBeamSpot*process.MeasurementTrackerEvent*process.ctftracksP5*process.siPixelClusterShapeCache)
    
    process.p = cms.Path(
                         ##### TRIGGER SELECTION #####
                         process.hltHighLevel*
                         process.scalersRawToDigi*
                         process.tcdsDigis*
                         process.APVPhases*
                         process.consecutiveHEs*
                         process.hltTriggerTypeFilter*
                         process.RecoForDQM_LocalReco*
                         process.DQMmodules*
                         process.RecoForDQM_TrkReco_cosmic*
                         process.siPixelPhase1OnlineDQM_source_cosmics*
                         process.siPixelPhase1OnlineDQM_harvesting
                         )
   
### pp/hi COLLISION SETTING

if (process.runType.getRunType() == process.runType.pp_run or process.runType.getRunType() == process.runType.pp_run_stage1 or process.runType.getRunType() == process.runType.hi_run):
    # Reco for pp collisions
    process.load('RecoTracker.IterativeTracking.InitialStepPreSplitting_cff')
    process.InitialStepPreSplittingTask.remove(process.initialStepTrackRefsForJetsPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.caloTowerForTrkPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.ak4CaloJetsForTrkPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.jetsForCoreTrackingPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.siPixelClusters)
    process.InitialStepPreSplittingTask.remove(process.siPixelRecHits)
    process.InitialStepPreSplittingTask.remove(process.MeasurementTrackerEvent)
    process.InitialStepPreSplittingTask.remove(process.siPixelClusterShapeCache)

    # Redefinition of siPixelClusters: has to be after RecoTracker.IterativeTracking.InitialStepPreSplitting_cff 
    from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters as _siPixelClusters
    process.siPixelClusters = _siPixelClusters.clone()

    from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
    process.PixelLayerTriplets.BPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
    process.PixelLayerTriplets.FPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
    from RecoTracker.PixelTrackFitting.PixelTracks_cff import *
    process.pixelTracksHitTriplets.SeedComparitorPSet.clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting'
    process.RecoForDQM_TrkReco = cms.Sequence(process.offlineBeamSpot*process.MeasurementTrackerEventPreSplitting*process.siPixelClusterShapeCachePreSplitting*process.recopixelvertexing*process.InitialStepPreSplitting)

    if (process.runType.getRunType() == process.runType.hi_run):
        #        process.SiPixelClusterSource.src = cms.InputTag("siPixelClustersPreSplitting")
        #        process.Reco = cms.Sequence(process.siPixelDigis*process.pixeltrackerlocalreco)
        process.Reco = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco)
    else:
        process.Reco = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco)
                          

    process.p = cms.Path(
      process.hltHighLevel #trigger selection
     *process.scalersRawToDigi
     *process.tcdsDigis
     *process.APVPhases
     *process.consecutiveHEs
     *process.Reco
     *process.siPixelClusters
     *process.DQMmodules
     *process.RecoForDQM_TrkReco
     *process.siPixelPhase1OnlineDQM_source_pprun
     *process.siPixelPhase1OnlineDQM_harvesting
    )

### FIXME: to add the HI Track Reconstruction    
### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print("Running with run type = ", process.runType.getRunType())
print("Final Source settings:", process.source)
