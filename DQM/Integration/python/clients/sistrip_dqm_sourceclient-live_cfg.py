from __future__ import print_function
import FWCore.ParameterSet.Config as cms

import sys
if 'runkey=hi_run' in sys.argv:
  from Configuration.Eras.Era_Run3_pp_on_PbPb_approxSiStripClusters_cff import Run3_pp_on_PbPb_approxSiStripClusters
  process = cms.Process("SiStripMonitor", Run3_pp_on_PbPb_approxSiStripClusters)
else:
  from Configuration.Eras.Era_Run3_cff import Run3
  process = cms.Process("SiStripMonitor", Run3)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siStripDigis',
                                         'siStripClusters',
                                         'siStripZeroSuppression',
                                         'SiStripClusterizer'),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
    destinations = cms.untracked.vstring('cout')
)

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
    process.load("DQM.Integration.config.unittestinputsource_cfi")
    from DQM.Integration.config.unittestinputsource_cfi import options
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
#from DQM.Integration.config.environment_cfi import HEAVYION

#process.runType.setRunType('cosmic_run')
#process.runType.setRunType('pp_run')

process.load("DQM.Integration.config.environment_cfi")
process.DQM.filter = '^(SiStrip|Tracking)(/[^/]+){0,5}$'

process.dqmEnv.subSystemFolder    = "SiStrip"
process.dqmSaver.tag = "SiStrip"
process.dqmSaver.backupLumiCount = 30
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = "SiStrip"
process.dqmSaverPB.runNumber = options.runNumber

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.dqmEnvTr = DQMEDAnalyzer('DQMEventInfo',
                 subSystemFolder = cms.untracked.string('Tracking'),
                 eventRateWindow = cms.untracked.double(0.5),
                 eventInfoFolder = cms.untracked.string('EventInfo')
)


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
    process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

#--------------------------------------------
## Patch to avoid using Run Info information in reconstruction
#

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

## Cosmic Track Reconstruction
if (process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1):
    process.load("RecoTracker.Configuration.RecoTrackerP5_cff")
    process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
else:
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
process.load("DQM.SiStripMonitorHardware.siStripFEDCheck_cfi")
process.siStripFEDCheck.RawDataTag = "rawDataCollector"
process.siStripFEDCheck.DirName    = 'SiStrip/FEDIntegrity_SM/'
process.siStripFEDCheck.doPLOTfedsPresent       = False # already produced by fedtest
process.siStripFEDCheck.doPLOTfedFatalErrors    = False # already produced by fedtest
process.siStripFEDCheck.doPLOTfedNonFatalErrors = False # already produced by fedtest
process.siStripFEDCheck.doPLOTnFEDinVsLS        = True
process.siStripFEDCheck.doPLOTnFEDinWdataVsLS   = True

#------------------------------
# Strip and Tracking DQM Source
#------------------------------
process.load("DQM.SiStripMonitorClient.SiStripSourceConfigP5_cff")
process.load("DQM.TrackingMonitorSource.TrackingSourceConfigP5_cff")
process.TrackMon_gentk.doLumiAnalysis = False
process.TrackMon_ckf.doLumiAnalysis = False
process.TrackMon_hi.doLumiAnalysis = False
process.TrackMon_ckf.AlgoName = 'CKFTk'

#--------------------------
# Quality Test
#--------------------------
from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.stripQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
    prescaleFactor = cms.untracked.int32(3),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndRun = cms.untracked.bool(True)
)

process.trackingQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/TrackingMonitorClient/data/tracking_qualitytest_config.xml'),
    prescaleFactor = cms.untracked.int32(3),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndRun = cms.untracked.bool(True)
)


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
process.hltHighLevel.throw =  False

#--------------------------
# Scheduling
#--------------------------
process.SiStripSources_LocalReco = cms.Sequence(process.siStripFEDMonitor*process.SiStripMonitorDigi*process.SiStripMonitorClusterReal)
if (process.runType.getRunType() == process.runType.commissioning_run):
    process.DQMCommon                = cms.Sequence(process.dqmEnv*process.dqmEnvTr*process.dqmSaver*process.dqmSaverPB)
else:
    process.DQMCommon                = cms.Sequence(process.stripQTester*process.trackingQTester*process.dqmEnv*process.dqmEnvTr*process.dqmSaver*process.dqmSaverPB)
if (process.runType.getRunType() == process.runType.hi_run):
    process.RecoForDQM_LocalReco     = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco)
else :
    process.RecoForDQM_LocalReco     = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.gtDigis*process.trackerlocalreco)
#------------------------------------------------------
# Switch for channel errors per FED ID trend plots.
#------------------------------------------------------
process.siStripFEDMonitor.fedErrorsVsIdVsLumiHistogramConfig.globalswitchon = True

#--------------------------
# Global Plot Switches
#--------------------------
process.SiStripMonitorDigi.TotalNumberOfDigisFailure.subdetswitchon = False

### COSMIC RUN SETTING
if (process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1):
    # event selection for cosmic data
    if ((process.runType.getRunType() == process.runType.cosmic_run) and live): process.source.SelectEvents = ['HLT*SingleMu*','HLT_L1*']
    # Reference run for cosmic
    # Source config for cosmic data
    process.SiStripSources_TrkReco_cosmic = cms.Sequence(process.SiStripMonitorTrack_ckf*process.TrackMon_ckf)
    # Client config for cosmic data
    ### STRIP
    process.load("DQM.SiStripMonitorClient.SiStripClientConfigP5_Cosmic_cff")
    process.SiStripAnalyserCosmic.RawDataTag = "rawDataCollector"
    process.SiStripAnalyserCosmic.TkMapCreationFrequency  = -1
    process.SiStripAnalyserCosmic.ShiftReportFrequency = -1
    process.SiStripAnalyserCosmic.StaticUpdateFrequency = 5
    process.SiStripAnalyserCosmic.MonitorSiStrip_PSet.MonitorSiStripBackPlaneCorrection = False
    process.SiStripClients           = cms.Sequence(process.SiStripAnalyserCosmic)
    ### TRACKING
    process.load("DQM.TrackingMonitorClient.TrackingClientConfigP5_Cosmic_cff")
    process.TrackingAnalyserCosmic.RawDataTag           = "rawDataCollector"
    process.TrackingAnalyserCosmic.ShiftReportFrequency = -1
    process.TrackingAnalyserCosmic.StaticUpdateFrequency = 5
    process.TrackingClient = cms.Sequence( process.TrackingAnalyserCosmic )

    # Reco for cosmic data
    process.load('RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cfi')
    process.simpleCosmicBONSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 450
    process.combinatorialcosmicseedfinderP5.MaxNumberOfCosmicClusters = 450

    process.RecoForDQM_TrkReco_cosmic = cms.Sequence(process.offlineBeamSpot*process.MeasurementTrackerEvent*process.ctftracksP5)

    process.stripQTester.qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_cosmic.xml')
    process.stripQTester.prescaleFactor          = 2
    process.stripQTester.getQualityTestsFromFile = True
    process.stripQTester.qtestOnEndLumi          = True
    process.stripQTester.qtestOnEndRun           = True

    process.trackingQTester.qtList                  = 'DQM/TrackingMonitorClient/data/tracking_qualitytest_config_cosmic.xml'
    process.trackingQTester.prescaleFactor          = 1
    process.trackingQTester.getQualityTestsFromFile = True
    process.trackingQTester.qtestOnEndLumi          = True
    process.trackingQTester.qtestOnEndRun           = True

    process.p = cms.Path(process.scalersRawToDigi*
                         process.tcdsDigis*
                         process.onlineMetaDataDigis*
                         process.APVPhases*
                         process.consecutiveHEs*
                         process.hltTriggerTypeFilter*
                         process.siStripFEDCheck *
                         process.RecoForDQM_LocalReco*
                         process.DQMCommon*
                         process.SiStripClients*
                         process.SiStripSources_LocalReco*
                         process.RecoForDQM_TrkReco_cosmic*
                         process.SiStripSources_TrkReco_cosmic*
                         process.TrackingClient
                         )


### COMMISSIONING RUN SETTINGS
if (process.runType.getRunType() == process.runType.commissioning_run):
    #event selection for commissioning runs
    if ((process.runType.getRunType() == process.runType.commissioning_run) and live):
        process.source.SelectEvents = ['HLT_*']

    process.SiStripFedMonitor = cms.Sequence(process.siStripFEDMonitor)
    process.p = cms.Path(
        process.siStripFEDCheck *
        process.SiStripFedMonitor *
        process.DQMCommon
    )

#else :
### pp COLLISION SETTING
if (process.runType.getRunType() == process.runType.pp_run or process.runType.getRunType() == process.runType.pp_run_stage1):
    #event selection for pp collisions
    if ((process.runType.getRunType() == process.runType.pp_run) and live):
        process.source.SelectEvents = [
            'HLT_L1*',
            'HLT_Jet*',
            'HLT_Physics*',
            'HLT_ZeroBias*',
            'HLT_PAL1*',
            'HLT_PAZeroBias*',
            'HLT_PAAK*'
            ]

    # Source and Client config for pp collisions

    process.SiStripMonitorDigi.UseDCSFiltering = False
    process.SiStripMonitorClusterReal.UseDCSFiltering = False

    process.MonitorTrackResiduals_gentk.Tracks                 = 'initialStepTracksPreSplitting'
    process.MonitorTrackResiduals_gentk.trajectoryInput        = 'initialStepTracksPreSplitting'
    process.TrackMon_gentk.TrackProducer    = 'initialStepTracksPreSplitting'
    process.TrackMon_gentk.allTrackProducer = 'initialStepTracksPreSplitting'
    process.SiStripMonitorTrack_gentk.TrackProducer = 'initialStepTracksPreSplitting'

    process.SiStripSources_TrkReco   = cms.Sequence(process.SiStripMonitorTrack_gentk*process.MonitorTrackResiduals_gentk*process.TrackMon_gentk)

    ### STRIP
    process.load("DQM.SiStripMonitorClient.SiStripClientConfigP5_cff")
    process.SiStripAnalyser.TkMapCreationFrequency  = -1
    process.SiStripAnalyser.ShiftReportFrequency = -1
    process.SiStripAnalyser.StaticUpdateFrequency = 5
    process.SiStripAnalyser.RawDataTag = "rawDataCollector"
    process.SiStripAnalyser.MonitorSiStrip_PSet.MonitorSiStripBackPlaneCorrection = False
    process.SiStripClients           = cms.Sequence(process.SiStripAnalyser)

    process.SiStripMonitorDigi.TotalNumberOfDigisFailure.integrateNLumisections = 25
    ### TRACKING
    process.load("DQM.TrackingMonitorClient.TrackingClientConfigP5_cff")
    process.TrackingAnalyser.ShiftReportFrequency = -1
    process.TrackingAnalyser.StaticUpdateFrequency = 5
    process.TrackingAnalyser.RawDataTag = "rawDataCollector"
    if offlineTesting :
        process.TrackingAnalyser.verbose = True
    process.TrackingClient = cms.Sequence( process.TrackingAnalyser )

    process.trackingQTester.qtList                  = 'DQM/TrackingMonitorClient/data/tracking_qualitytest_config.xml'
    process.trackingQTester.prescaleFactor          = 1
    process.trackingQTester.getQualityTestsFromFile = True
    process.trackingQTester.qtestOnEndLumi          = True
    process.trackingQTester.qtestOnEndRun           = True

    # Reco for pp collisions

    process.load('RecoTracker.IterativeTracking.InitialStepPreSplitting_cff')
    '''process.InitialStepPreSplitting.remove(process.initialStepTrackRefsForJetsPreSplitting)
    process.InitialStepPreSplitting.remove(process.caloTowerForTrkPreSplitting)
    process.InitialStepPreSplitting.remove(process.ak4CaloJetsForTrkPreSplitting)
    process.InitialStepPreSplitting.remove(process.jetsForCoreTrackingPreSplitting)
    process.InitialStepPreSplitting.remove(process.siPixelClusters)
    process.InitialStepPreSplitting.remove(process.siPixelRecHits)
    process.InitialStepPreSplitting.remove(process.MeasurementTrackerEvent)
    process.InitialStepPreSplitting.remove(process.siPixelClusterShapeCache)'''

    process.InitialStepPreSplittingTask.remove(process.initialStepTrackRefsForJetsPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.caloTowerForTrkPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.ak4CaloJetsForTrkPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.jetsForCoreTrackingPreSplitting)
    process.InitialStepPreSplittingTask.remove(process.siPixelClusters)
    process.InitialStepPreSplittingTask.remove(process.siPixelRecHits)
    process.InitialStepPreSplittingTask.remove(process.MeasurementTrackerEvent)

    # Redefinition of siPixelClusters: has to be after RecoTracker.IterativeTracking.InitialStepPreSplitting_cff
    process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

    from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
    process.PixelLayerTriplets.BPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
    process.PixelLayerTriplets.FPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
    from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
    process.pixelTracksHitTriplets.SeedComparitorPSet.clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting'

    process.RecoForDQM_TrkReco = cms.Sequence(process.offlineBeamSpot*process.MeasurementTrackerEventPreSplitting*process.siPixelClusterShapeCachePreSplitting*process.recopixelvertexing*process.InitialStepPreSplitting)

    process.p = cms.Path(
        process.scalersRawToDigi*
        process.tcdsDigis*
        process.onlineMetaDataDigis*
        process.APVPhases*
        process.consecutiveHEs*
        process.hltTriggerTypeFilter*
        process.siStripFEDCheck *
        process.RecoForDQM_LocalReco*
        process.siPixelClusters*
        process.DQMCommon*
        process.SiStripClients*
        process.SiStripSources_LocalReco*
        ##### TRIGGER SELECTION #####
        process.hltHighLevel*
        process.RecoForDQM_TrkReco*
        process.SiStripSources_TrkReco*
        process.TrackingClient
        )

#--------------------------------------------------
# For high PU run - no tracking in cmssw42x
#--------------------------------------------------
if (process.runType.getRunType() == process.runType.hpu_run):

    # Simple filter for event
    # 2012.07.09 highPU fill should have /cdaq/special/HighPUFill/July2012/HLT/V6 as trigger table
    # where HLT_ZeroBias in the DQM stream has ~50Hz
    # the expected reconstruction time should be ~ several seconds
    # => PRESCALE = 50
    # but try firstly w/ 30, maybe it is enough
    process.eventFilter.EventsToSkip = 30

    # change the HLT trigger path selection
    # it should already be ok, but the name could be changed
    process.hltHighLevel.HLTPaths = ['HLT_ZeroBias*'] 

 #        process.DQMEventStreamerReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('HLT_600Tower*','HLT_L1*','HLT_Jet*','HLT_HT*','HLT_MinBias_*','HLT_Physics*', 'HLT_ZeroBias*'))
#

    process.SiStripMonitorDigi.UseDCSFiltering = False
    process.SiStripMonitorClusterReal.UseDCSFiltering = False

    process.MonitorTrackResiduals_gentk.Tracks                 = 'earlyGeneralTracks'
    process.MonitorTrackResiduals_gentk.trajectoryInput        = 'earlyGeneralTracks'
    process.TrackMon_gentk.TrackProducer          = "earlyGeneralTracks"
    process.TrackMon_gentk.allTrackProducer = "earlyGeneralTracks"
    process.SiStripMonitorTrack_gentk.TrackProducer    = 'earlyGeneralTracks'

    process.SiStripSources_TrkReco   = cms.Sequence(process.SiStripMonitorTrack_gentk*process.MonitorTrackResiduals_gentk*process.TrackMon_gentk)
    process.load("DQM.SiStripMonitorClient.SiStripClientConfigP5_cff")
    process.SiStripAnalyser.TkMapCreationFrequency  = -1
    process.SiStripAnalyser.ShiftReportFrequency = -1
    process.SiStripAnalyser.StaticUpdateFrequency = 5
    process.SiStripAnalyser.RawDataTag = "rawDataCollector"
    process.SiStripAnalyser.MonitorSiStrip_PSet.MonitorSiStripBackPlaneCorrection = False
    process.SiStripClients           = cms.Sequence(process.SiStripAnalyser)
    ### TRACKING
    process.load("DQM.TrackingMonitorClient.TrackingClientConfigP5_cff")
    process.TrackingAnalyser.ShiftReportFrequency = -1
    process.TrackingAnalyser.StaticUpdateFrequency = 5
    process.TrackingAnalyser.RawDataTag = cms.untracked.InputTag("rawDataCollector")
    process.TrackingClient = cms.Sequence( process.TrackingAnalyser )

    # Reco for pp collisions

    process.load('RecoTracker.Configuration.RecoTracker_cff')

    #process.newCombinedSeeds.seedCollections = cms.VInputTag(
    #    cms.InputTag('initialStepSeeds'),
    #    )

    process.load('RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff')
    import RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi
    process.load('RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi')
    process.earlyGeneralTracks.TrackProducers = (
        cms.InputTag('initialStepTracks'),
        )

    process.earlyGeneralTracks.hasSelector=cms.vint32(1)
    process.earlyGeneralTracks.selectedTrackQuals = cms.VInputTag(
#        cms.InputTag("initialStepSelector","initialStep"),
        cms.InputTag("initialStep"),
        )
    process.earlyGeneralTracks.setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0), pQual=cms.bool(True) ) )

    process.load("RecoTracker.IterativeTracking.iterativeTk_cff")

    process.iterTracking_FirstStep =cms.Sequence(
        process.InitialStep
        *process.earlyGeneralTracks
        )

    process.RecoForDQM_TrkReco       = cms.Sequence(process.offlineBeamSpot*process.MeasurementTrackerEvent*process.siPixelClusterShapeCache*process.recopixelvertexing*process.iterTracking_FirstStep)

    process.p = cms.Path(process.scalersRawToDigi*
                         process.tcdsDigis*
                         process.onlineMetaDataDigis*
                         process.APVPhases*
                         process.consecutiveHEs*
                         process.hltTriggerTypeFilter*
                         process.siStripFEDCheck *
                         process.RecoForDQM_LocalReco*
                         process.DQMCommon*
                         process.SiStripClients*
                         process.SiStripSources_LocalReco*
                         process.hltHighLevel*
                         process.eventFilter*
                         process.RecoForDQM_TrkReco*
                         process.SiStripSources_TrkReco*
                         process.TrackingClient
                         )

process.castorDigis.InputLabel = "rawDataCollector"
process.csctfDigis.producer = "rawDataCollector"
process.dttfDigis.DTTF_FED_Source = "rawDataCollector"
process.ecalDigis.cpu.InputLabel = "rawDataCollector"
process.ecalPreshowerDigis.sourceTag = "rawDataCollector"
process.gctDigis.inputLabel = "rawDataCollector"
process.gtDigis.DaqGtInputTag = "rawDataCollector"
process.hcalDigis.InputLabel = "rawDataCollector"
process.muonCSCDigis.InputObjects = "rawDataCollector"
process.muonDTDigis.inputLabel = "rawDataCollector"
process.muonRPCDigis.InputLabel = "rawDataCollector"
process.scalersRawToDigi.scalersInputTag = "rawDataCollector"
process.siPixelDigis.cpu.InputLabel = "rawDataCollector"
process.siStripDigis.ProductLabel = "rawDataCollector"
process.siStripFEDMonitor.RawDataTag = "rawDataCollector"
#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print("Running with run type = ", process.runType.getRunType())
### HEAVY ION SETTING
if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = "rawDataRepacker"
    process.csctfDigis.producer = "rawDataRepacker"
    process.dttfDigis.DTTF_FED_Source = "rawDataRepacker"
    process.ecalDigis.cpu.InputLabel = "rawDataRepacker"
    process.ecalPreshowerDigis.sourceTag = "rawDataRepacker"
    process.gctDigis.inputLabel = "rawDataRepacker"
    process.hcalDigis.InputLabel = "rawDataRepacker"
    process.muonCSCDigis.InputObjects = "rawDataRepacker"
    process.muonDTDigis.inputLabel = "rawDataRepacker"
    process.muonRPCDigis.InputLabel = "rawDataRepacker"
    process.scalersRawToDigi.scalersInputTag = "rawDataRepacker"
    process.siPixelDigis.cpu.InputLabel = "rawDataRepacker"
    process.siStripDigis.ProductLabel = "rawDataRepacker"
    process.siStripFEDMonitor.RawDataTag = "rawDataRepacker"

    if ((process.runType.getRunType() == process.runType.hi_run) and live):
        process.source.SelectEvents = [
            'HLT_HICentralityVeto*',
#            'HLT_HIMinimumBias*',
#            'HLT_HIZeroBias*'
            'HLT_HIPhysics*'
            ]

    process.SiStripMonitorDigi.UseDCSFiltering = False
    process.SiStripMonitorClusterReal.UseDCSFiltering = False

    process.MonitorTrackResiduals_gentk.Tracks                 = 'initialStepTracksPreSplitting'
    process.MonitorTrackResiduals_gentk.trajectoryInput        = 'initialStepTracksPreSplitting'
    process.TrackMon_gentk.TrackProducer    = 'initialStepTracksPreSplitting'
    process.TrackMon_gentk.allTrackProducer = 'initialStepTracksPreSplitting'
    process.SiStripMonitorTrack_gentk.TrackProducer = 'initialStepTracksPreSplitting'

    process.SiStripSources_TrkReco   = cms.Sequence(process.SiStripMonitorTrack_gentk*process.MonitorTrackResiduals_gentk*process.TrackMon_gentk)

    ### STRIP
    process.load("DQM.SiStripMonitorClient.SiStripClientConfigP5_cff")
    process.SiStripAnalyser.TkMapCreationFrequency  = -1
    process.SiStripAnalyser.ShiftReportFrequency = -1
    process.SiStripAnalyser.StaticUpdateFrequency = 5
    process.SiStripAnalyser.RawDataTag = "rawDataRepacker"
    process.SiStripAnalyser.MonitorSiStrip_PSet.MonitorSiStripBackPlaneCorrection = False
    process.SiStripClients           = cms.Sequence(process.SiStripAnalyser)

    process.SiStripMonitorDigi.TotalNumberOfDigisFailure.integrateNLumisections = 25
    ### TRACKING
    process.load("DQM.TrackingMonitorClient.TrackingClientConfigP5_cff")
    process.TrackingAnalyser.ShiftReportFrequency = -1
    process.TrackingAnalyser.StaticUpdateFrequency = 5
    process.TrackingAnalyser.RawDataTag = "rawDataRepacker"
    if offlineTesting :
        process.TrackingAnalyser.verbose = True
    process.TrackingClient = cms.Sequence( process.TrackingAnalyser )

    process.stripQTester.qtList = 'DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_heavyion.xml'
    process.stripQTester.prescaleFactor          = 2
    process.stripQTester.getQualityTestsFromFile = True
    process.stripQTester.qtestOnEndLumi          = True
    process.stripQTester.qtestOnEndRun           = True

    process.trackingQTester.qtList                  = 'DQM/TrackingMonitorClient/data/tracking_qualitytest_config_heavyion.xml'
    process.trackingQTester.prescaleFactor          = 1
    process.trackingQTester.getQualityTestsFromFile = True
    process.trackingQTester.qtestOnEndLumi          = True
    process.trackingQTester.qtestOnEndRun           = True

    # Reco for pp collisions

    process.load('RecoTracker.IterativeTracking.InitialStepPreSplitting_cff')
    '''process.InitialStepPreSplitting.remove(process.initialStepTrackRefsForJetsPreSplitting)
    process.InitialStepPreSplitting.remove(process.caloTowerForTrkPreSplitting)
    process.InitialStepPreSplitting.remove(process.ak4CaloJetsForTrkPreSplitting)
    process.InitialStepPreSplitting.remove(process.jetsForCoreTrackingPreSplitting)
    process.InitialStepPreSplitting.remove(process.siPixelClusters)
    process.InitialStepPreSplitting.remove(process.siPixelRecHits)
    process.InitialStepPreSplitting.remove(process.MeasurementTrackerEvent)
    process.InitialStepPreSplitting.remove(process.siPixelClusterShapeCache)'''

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
        inputTag  = 'siPixelClusters',
        minClusters = 1,
        maxClusters = 50000
        )

    # BaselineValidator Module
    from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import siStripDigis as _siStripDigis
    process.siStripDigisNoZS=_siStripDigis.clone(
        ProductLabel = "rawDataCollector")
    process.SiStripBaselineValidator.srcProcessedRawDigi =  'siStripDigisNoZS:ZeroSuppressed'


    from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
    process.PixelLayerTriplets.BPix.HitProducer = 'siPixelRecHitsPreSplitting'
    process.PixelLayerTriplets.FPix.HitProducer = 'siPixelRecHitsPreSplitting'
    from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
    process.pixelTracksHitTriplets.SeedComparitorPSet.clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting'

    process.RecoForDQM_TrkReco = cms.Sequence(process.offlineBeamSpot*process.MeasurementTrackerEventPreSplitting*process.siPixelClusterShapeCachePreSplitting*process.recopixelvertexing*process.InitialStepPreSplitting)

    process.p = cms.Path(
        process.scalersRawToDigi*
        process.tcdsDigis*
        process.onlineMetaDataDigis*
        process.APVPhases*
        process.consecutiveHEs*
        process.hltTriggerTypeFilter*
        process.siStripFEDCheck *
        process.siStripDigisNoZS*
        process.SiStripBaselineValidator*
        process.RecoForDQM_LocalReco*
        process.siPixelClusters*
        process.DQMCommon*
        process.SiStripClients*
        process.SiStripSources_LocalReco*
        process.multFilter*
        ##### TRIGGER SELECTION #####
        process.hltHighLevel*
        process.RecoForDQM_TrkReco*
        process.SiStripSources_TrkReco*
        process.TrackingClient
    )

    # append the approximate clusters monitoring for the HI run case
    from DQM.SiStripMonitorApproximateCluster.SiStripMonitorApproximateCluster_cfi import SiStripMonitorApproximateCluster
    process.siStripApproximateClusterComparator = SiStripMonitorApproximateCluster.clone(compareClusters = True,
                                                                                         ClustersProducer = "hltSiStripClusterizerForRawPrime")
    process.p.insert(process.p.index(process.TrackingClient)+1,process.siStripApproximateClusterComparator)

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
print("Final Source settings:", process.source)
