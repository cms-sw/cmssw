from __future__ import print_function
import FWCore.ParameterSet.Config as cms

import sys
if 'runkey=hi_run' in sys.argv:
  from Configuration.Eras.Era_Run3_pp_on_PbPb_cff import Run3_pp_on_PbPb
  process = cms.Process("SiStripApproxMonitor", Run3_pp_on_PbPb)
else:
  from Configuration.Eras.Era_Run3_cff import Run3
  process = cms.Process("SiStripApproxMonitor", Run3)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = cms.untracked.vstring('siStripDigis',
                                                           'siStripClusters',
                                                           'siStripZeroSuppression',
                                                           'SiStripClusterizer',
                                                           'siStripApproximateClusterComparator')
process.MessageLogger.cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR'))

live=True
unitTest=False

if 'unitTest=True' in sys.argv:
    live=False
    unitTest=True

# uncomment for running on lxplus
#live=False
offlineTesting=not live
#print "live: "+str(live)+" ==> offlineTesting: "+str(offlineTesting)

#----------------------------
# Event Source
#-----------------------------
# for live online DQM in P5
if (unitTest):
  process.load("DQM.Integration.config.unitteststreamerinputsource_cfi")
  from DQM.Integration.config.unitteststreamerinputsource_cfi import options
elif (live):
  process.load("DQM.Integration.config.inputsource_cfi")
  from DQM.Integration.config.inputsource_cfi import options
# for testing in lxplus
elif(offlineTesting):
  process.load("DQM.Integration.config.fileinputsource_cfi")
  from DQM.Integration.config.fileinputsource_cfi import options

#----------------------------
# DQM Live Environment
#-----------------------------

process.load("DQM.Integration.config.environment_cfi")
process.DQM.filter = '^(SiStripApproximateClusters)(/[^/]+){0,5}$'

process.dqmEnv.subSystemFolder    = "SiStripApproximateClusters"
process.dqmSaver.tag = "SiStripApproximateClusters"
process.dqmSaver.backupLumiCount = 30
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = "SiStripApproximateClusters"
process.dqmSaverPB.runNumber = options.runNumber

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.dqmEnvTr = DQMEDAnalyzer('DQMEventInfo',
                                 subSystemFolder = cms.untracked.string('SiStripApproximateClusters'),
                                 eventRateWindow = cms.untracked.double(0.5),
                                 eventInfoFolder = cms.untracked.string('EventInfo'))

#-----------------------------
# Magnetic Field
#-----------------------------
process.load('Configuration.StandardSequences.MagneticField_cff')

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
    #you may need to set manually the GT in the line below
    process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_hlt', '')


print("Will process with GlobalTag %s",process.GlobalTag.globaltag.value())

#--------------------------------------------
# Patch to avoid using Run Info information in reconstruction
#--------------------------------------------
process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
   cms.PSet( record = cms.string("SiStripDetVOffRcd"),    tag    = cms.string("") ),
   cms.PSet( record = cms.string("SiStripDetCablingRcd"), tag    = cms.string("") ),
   cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
   cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") ),
   cms.PSet( record = cms.string("SiStripBadModuleRcd"),  tag    = cms.string("") )
   )
#-------------------------------------------

#-----------------------
#  Reconstruction Modules
#-----------------------
## Collision Reconstruction
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

import RecoVertex.BeamSpotProducer.onlineBeamSpotESProducer_cfi as _mod
process.BeamSpotESProducer = _mod.onlineBeamSpotESProducer.clone()

# for running offline enhance the time validity of the online beamspot in DB
if ((not live) or process.isDqmPlayback.value): 
  process.BeamSpotESProducer.timeThreshold = cms.int32(int(1e6))

import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

#
# Strip FED check
#
rawDataCollectorLabel = 'rawDataCollector'

#--------------------------
# Service
#--------------------------
process.AdaptorConfig = cms.Service("AdaptorConfig")

# Simple filter for event
process.eventFilter = cms.EDFilter("SimpleEventFilter",
#                   EventsToSkip = cms.untracked.int32(3)
                   EventsToSkip = cms.untracked.int32(100)
)

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
process.hltLevel1GTSeed.L1TechTriggerSeeding = True
process.hltLevel1GTSeed.L1SeedsLogicalExpression = 'NOT (36 OR 37 OR 38 OR 39)'

# HLT trigger selection (HLT_ZeroBias)
# modified for 0 Tesla HLT menu (no ZeroBias_*)
process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
if (process.runType.getRunType() == process.runType.hi_run):
    #--------------------------
    # HI Runs HLT path
    #--------------------------
    process.hltHighLevel.HLTPaths = ['HLT_ZeroBias_*' , 'HLT_HIZeroBias*' , 'HLT_ZeroBias1_*' , 'HLT_PAZeroBias_*' , 'HLT_PAZeroBias1_*', 'HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_*' , 'HLT_HICentralityVeto*','HLT_HIMinimumBias*', 'HLT_HIPhysics*']
else:
    process.hltHighLevel.HLTPaths = ['HLT_ZeroBias_*' , 'HLT_ZeroBias1_*' , 'HLT_PAZeroBias_*' , 'HLT_PAZeroBias1_*', 'HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_*']
process.hltHighLevel.andOr = True
process.hltHighLevel.throw = False

if (process.runType.getRunType() == process.runType.hi_run):
    process.RecoForDQM_LocalReco     = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco)
else :
    process.RecoForDQM_LocalReco     = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.gtDigis*process.trackerlocalreco)

process.DQMCommon = cms.Sequence(process.dqmEnv*process.dqmEnvTr*process.dqmSaver*process.dqmSaverPB)

print("Running with run type = ", process.runType.getRunTypeName())

### HEAVY ION SETTING
if process.runType.getRunType() == process.runType.hi_run:
    rawDataRepackerLabel = 'rawDataRepacker'
    process.castorDigis.InputLabel = rawDataRepackerLabel
    process.csctfDigis.producer = rawDataRepackerLabel
    process.dttfDigis.DTTF_FED_Source = rawDataRepackerLabel
    process.ecalDigisCPU.InputLabel = rawDataRepackerLabel
    process.ecalPreshowerDigis.sourceTag = rawDataRepackerLabel
    process.gctDigis.inputLabel = rawDataRepackerLabel
    process.hcalDigis.InputLabel = rawDataRepackerLabel
    process.muonCSCDigis.InputObjects = rawDataRepackerLabel
    process.muonDTDigis.inputLabel = rawDataRepackerLabel
    process.muonRPCDigis.InputLabel = rawDataRepackerLabel
    process.scalersRawToDigi.scalersInputTag = rawDataRepackerLabel
    process.siPixelDigis.cpu.InputLabel = rawDataRepackerLabel
    process.siStripDigis.ProductLabel = rawDataRepackerLabel

    if ((process.runType.getRunType() == process.runType.hi_run) and live):
        process.source.SelectEvents = [
#            'HLT_HICentralityVeto*', # present in 2018 and 2022 HIon menus
            'HLT_HIMinimumBias*',     # replaced HLT_HICentralityVeto starting from the 2023 HIon menu
#            'HLT_HIZeroBias*',       # present in DQM stream of HIon menu, but not used in this client
            'HLT_HIPhysics*'
        ]

    process.load('RecoTracker.IterativeTracking.InitialStepPreSplitting_cff')
    process.InitialStepPreSplittingTask.remove(process.initialStepTrackRefsForJetsPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.caloTowerForTrkPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.ak4CaloJetsForTrkPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.jetsForCoreTrackingPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.siPixelClusters)
    process.InitialStepPreSplittingTask.remove(process.siPixelRecHits)
    process.InitialStepPreSplittingTask.remove(process.MeasurementTrackerEvent)

    # Redefinition of siPixelClusters: has to be after RecoTracker.IterativeTracking.InitialStepPreSplitting_cff
    process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

    # Select events based on the pixel cluster multiplicity
    import  HLTrigger.special.hltPixelActivityFilter_cfi
    process.multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
        inputTag = 'siPixelClusters',
        minClusters = 1,
        maxClusters = 50000
    )

    from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
    process.PixelLayerTriplets.BPix.HitProducer = 'siPixelRecHitsPreSplitting'
    process.PixelLayerTriplets.FPix.HitProducer = 'siPixelRecHitsPreSplitting'
    from RecoTracker.PixelTrackFitting.PixelTracks_cff import *
    process.pixelTracksHitTriplets.SeedComparitorPSet.clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting'

    process.RecoForDQM_TrkReco = cms.Sequence(
        process.offlineBeamSpot
      * process.MeasurementTrackerEventPreSplitting
      * process.siPixelClusterShapeCachePreSplitting
      * process.recopixelvertexing
      * process.InitialStepPreSplitting
    )

    # append the approximate clusters monitoring for the HI run case
    from DQM.SiStripMonitorApproximateCluster.SiStripMonitorApproximateCluster_cfi import SiStripMonitorApproximateCluster
    process.siStripApproximateClusterComparator = SiStripMonitorApproximateCluster.clone(
        compareClusters = True,
        ClustersProducer = "siStripClusters"
    )

    process.p = cms.Path(
        process.scalersRawToDigi*
        process.tcdsDigis*
        process.onlineMetaDataDigis*
        process.APVPhases*
        process.consecutiveHEs*
        process.hltTriggerTypeFilter*
        process.RecoForDQM_LocalReco*
        process.siPixelClusters*
        process.DQMCommon*
        process.multFilter*
        ##### TRIGGER SELECTION #####
        process.hltHighLevel*
        process.RecoForDQM_TrkReco*
        process.siStripApproximateClusterComparator
    )

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
print("Final Source settings:", process.source)
