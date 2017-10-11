import FWCore.ParameterSet.Config as cms


from Configuration.StandardSequences.Eras import eras


process = cms.Process("SiStrpDQMLive",eras.Run2_2017_pp_on_XeXe)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siStripDigis', 
                                         'siStripClusters', 
                                         'siStripZeroSuppression', 
                                        'SiStripClusterizer'),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
    destinations = cms.untracked.vstring('cout')
)

live=True
# uncomment for running on lxplus
#live=False
offlineTesting=not live
#print "live: "+str(live)+" ==> offlineTesting: "+str(offlineTesting) 

#----------------------------
# Event Source
#-----------------------------
# for live online DQM in P5
if (live):
    process.load("DQM.Integration.config.inputsource_cfi")
# for testing in lxplus
elif(offlineTesting):
    process.load("DQM.Integration.config.fileinputsource_cfi")

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

process.dqmEnvTr = cms.EDAnalyzer("DQMEventInfo",
                 subSystemFolder = cms.untracked.string('Tracking'),
                 eventRateWindow = cms.untracked.double(0.5),
                 eventInfoFolder = cms.untracked.string('EventInfo')
)

## uncooment for running in local
## collector
#process.DQM.collectorHost = 'vmepcs2b18-20.cms'
#process.DQM.collectorPort = 9190
#process.DQM.collectorHost = 'lxplus414'
#process.DQM.collectorPort = 8070

#--------------------------
#  Lumi Producer and DB access
#-------------------------
#process.DBService=cms.Service('DBService',
#                              authPath= cms.untracked.string('/nfshome0/popcondev/conddb/')
#                              )
#process.DIPLumiProducer=cms.ESSource("DIPLumiProducer",
#          connect=cms.string('oracle://cms_omds_lb/CMS_RUNTIME_LOGGER')
#                            )

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
    from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
    process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

#--------------------------------------------
## Patch to avoid using Run Info information in reconstruction
#
process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
   cms.PSet( record = cms.string("SiStripDetVOffRcd"),    tag    = cms.string("") ),
   cms.PSet( record = cms.string("SiStripDetCablingRcd"), tag    = cms.string("") ),
#  cms.PSet( record = cms.string("RunInfoRcd"),           tag    = cms.string("") ),
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
#process.siStripDigis.UnpackBadChannels = cms.bool(True)

## Cosmic Track Reconstruction
#if (process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1):
#    process.load("RecoTracker.Configuration.RecoTrackerP5_cff")
#    process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
#else:
process.load("Configuration.StandardSequences.Reconstruction_cff")


## # offline beam spot
## process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
## # online beam spot
## process.load("RecoVertex.BeamSpotProducer.BeamSpotOnline_cff")

# handle onlineBeamSpot w/o changing all configuration
# the same shortcut is also used in Express ;)
# http://cmslxr.fnal.gov/lxr/source/Configuration/DataProcessing/python/RecoTLR.py#044
import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

#
# Strip FED check
#
process.load("DQM.SiStripMonitorHardware.siStripFEDCheck_cfi")
process.siStripFEDCheck.RawDataTag = cms.InputTag("rawDataCollector")
process.siStripFEDCheck.DirName    = cms.untracked.string('SiStrip/FEDIntegrity_SM/')
process.siStripFEDCheck.doPLOTfedsPresent       = cms.bool(False) # already produced by fedtest
process.siStripFEDCheck.doPLOTfedFatalErrors    = cms.bool(False) # already produced by fedtest
process.siStripFEDCheck.doPLOTfedNonFatalErrors = cms.bool(False) # already produced by fedtest
process.siStripFEDCheck.doPLOTnFEDinVsLS        = cms.bool(True)
process.siStripFEDCheck.doPLOTnFEDinWdataVsLS   = cms.bool(True)

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
process.stripQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
    prescaleFactor = cms.untracked.int32(3),                               
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndRun = cms.untracked.bool(True)
)

process.trackingQTester = cms.EDAnalyzer("QualityTester",
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
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('NOT (36 OR 37 OR 38 OR 39)')

# HLT trigger selection (HLT_ZeroBias)
# modified for 0 Tesla HLT menu (no ZeroBias_*)
process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
process.hltHighLevel.HLTPaths = cms.vstring( 'HLT_ZeroBias_*' , 'HLT_HIZeroBias*', 'HLT_ZeroBias1_*' , 'HLT_PAZeroBias_*' , 'HLT_PAZeroBias1_*', 'HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_*','HLT_HICentralityVeto*','HLT_HIMinBias*')
process.hltHighLevel.andOr = cms.bool(True)
process.hltHighLevel.throw =  cms.bool(False)

#--------------------------
# Scheduling
#--------------------------
process.SiStripSources_LocalReco = cms.Sequence(process.siStripFEDMonitor*process.SiStripMonitorDigi*process.SiStripMonitorClusterReal)
process.DQMCommon                = cms.Sequence(process.stripQTester*process.trackingQTester*process.dqmEnv*process.dqmEnvTr*process.dqmSaver)
process.RecoForDQM_LocalReco     = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco)

#--------------------------
# Global Plot Switches
#--------------------------
process.SiStripMonitorDigi.TotalNumberOfDigisFailure.subdetswitchon = cms.bool(False)

#
#### COSMIC RUN SETTING
#if (process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1):
#    # event selection for cosmic data
#    if (process.runType.getRunType() == process.runType.cosmic_run): process.source.SelectEvents = cms.untracked.vstring('HLT*SingleMu*','HLT_L1*')
#    # Reference run for cosmic
#    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/sistrip_reference_cosmic.root'
#    # Source config for cosmic data
#    process.SiStripSources_TrkReco_cosmic = cms.Sequence(process.SiStripMonitorTrack_ckf*process.TrackMon_ckf)
#    # Client config for cosmic data
#    ### STRIP
#    process.load("DQM.SiStripMonitorClient.SiStripClientConfigP5_Cosmic_cff")
#    process.SiStripAnalyserCosmic.RawDataTag = cms.untracked.InputTag("rawDataCollector")
#    process.SiStripAnalyserCosmic.TkMapCreationFrequency  = -1
#    process.SiStripAnalyserCosmic.ShiftReportFrequency = -1
#    process.SiStripAnalyserCosmic.StaticUpdateFrequency = 5
#    process.SiStripAnalyserCosmic.MonitorSiStripBackPlaneCorrection = cms.bool(False)
#    process.SiStripClients           = cms.Sequence(process.SiStripAnalyserCosmic)
#    ### TRACKING 
#    process.load("DQM.TrackingMonitorClient.TrackingClientConfigP5_Cosmic_cff")
#    process.TrackingAnalyserCosmic.RawDataTag           = cms.untracked.InputTag("rawDataCollector")
#    process.TrackingAnalyserCosmic.ShiftReportFrequency = -1
#    process.TrackingAnalyserCosmic.StaticUpdateFrequency = 5
#    process.TrackingClient = cms.Sequence( process.TrackingAnalyserCosmic )
#
#    # Reco for cosmic data
#    process.load('RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cfi')
#    process.simpleCosmicBONSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 450
#    process.combinatorialcosmicseedfinderP5.MaxNumberOfCosmicClusters = 450
#
#    process.RecoForDQM_TrkReco_cosmic = cms.Sequence(process.offlineBeamSpot*process.MeasurementTrackerEvent*process.ctftracksP5)
#
#    process.stripQTester.qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_cosmic.xml')
#    process.stripQTester.prescaleFactor          = cms.untracked.int32(2)
#    process.stripQTester.getQualityTestsFromFile = cms.untracked.bool(True)
#    process.stripQTester.qtestOnEndLumi          = cms.untracked.bool(True)
#    process.stripQTester.qtestOnEndRun           = cms.untracked.bool(True)
#
#    process.trackingQTester.qtList                  = cms.untracked.FileInPath('DQM/TrackingMonitorClient/data/tracking_qualitytest_config_cosmic.xml')
#    process.trackingQTester.prescaleFactor          = cms.untracked.int32(1)
#    process.trackingQTester.getQualityTestsFromFile = cms.untracked.bool(True)
#    process.trackingQTester.qtestOnEndLumi          = cms.untracked.bool(True)
#    process.trackingQTester.qtestOnEndRun           = cms.untracked.bool(True)                                                                    
#
#    process.p = cms.Path(process.scalersRawToDigi*
#                         process.APVPhases*
#                         process.consecutiveHEs*
#                         process.hltTriggerTypeFilter*
#                         process.siStripFEDCheck *
#                         process.RecoForDQM_LocalReco*
#                         process.DQMCommon*
#                         process.SiStripClients*
#                         process.SiStripSources_LocalReco*
#                         process.RecoForDQM_TrkReco_cosmic*
#                         process.SiStripSources_TrkReco_cosmic*
#                         process.TrackingClient
#                         )
#
#
#
#else :
### pp COLLISION SETTING
process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
process.siStripFEDMonitor.RawDataTag = cms.untracked.InputTag("rawDataRepacker")

    #event selection for pp collisions
  
process.source.SelectEvents = cms.untracked.vstring(
           'HLT_HICentralityVeto*','HLT_HIMinBias*','HLT_HIZeroBias*'
            )

process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/sistrip_reference_pp.root'
    # Source and Client config for pp collisions

process.SiStripMonitorDigi.UseDCSFiltering = cms.bool(False)
process.SiStripMonitorClusterReal.UseDCSFiltering = cms.bool(False)
    
process.MonitorTrackResiduals_gentk.Tracks                 = 'initialStepTracksPreSplitting'
process.MonitorTrackResiduals_gentk.trajectoryInput        = 'initialStepTracksPreSplitting'
process.MonitorTrackResiduals_gentk.TrackProducer          = cms.string('initialStepTracksPreSplitting')
process.TrackMon_gentk.TrackProducer    = cms.InputTag('initialStepTracksPreSplitting')
process.TrackMon_gentk.allTrackProducer = cms.InputTag('initialStepTracksPreSplitting')
process.SiStripMonitorTrack_gentk.TrackProducer = 'initialStepTracksPreSplitting'

process.SiStripSources_TrkReco   = cms.Sequence(process.SiStripMonitorTrack_gentk*process.MonitorTrackResiduals_gentk*process.TrackMon_gentk)

### STRIP
process.load("DQM.SiStripMonitorClient.SiStripClientConfigP5_cff")
process.SiStripAnalyser.UseGoodTracks  = cms.untracked.bool(True)
process.SiStripAnalyser.TkMapCreationFrequency  = -1
process.SiStripAnalyser.ShiftReportFrequency = -1
process.SiStripAnalyser.StaticUpdateFrequency = 5
process.SiStripAnalyser.RawDataTag = cms.untracked.InputTag("rawDataRepacker")
process.SiStripAnalyser.MonitorSiStripBackPlaneCorrection = cms.bool(False)
process.SiStripClients           = cms.Sequence(process.SiStripAnalyser)

process.SiStripMonitorDigi.TotalNumberOfDigisFailure.integrateNLumisections = cms.int32(25)
### TRACKING 
process.load("DQM.TrackingMonitorClient.TrackingClientConfigP5_cff")
process.TrackingAnalyser.ShiftReportFrequency = -1
process.TrackingAnalyser.StaticUpdateFrequency = 5
process.TrackingAnalyser.RawDataTag = cms.untracked.InputTag("rawDataRepacker")

process.TrackingClient = cms.Sequence( process.TrackingAnalyser )
    
process.trackingQTester.qtList                  = cms.untracked.FileInPath('DQM/TrackingMonitorClient/data/tracking_qualitytest_config.xml')
process.trackingQTester.prescaleFactor          = cms.untracked.int32(1)
process.trackingQTester.getQualityTestsFromFile = cms.untracked.bool(True)
process.trackingQTester.qtestOnEndLumi          = cms.untracked.bool(True)
process.trackingQTester.qtestOnEndRun           = cms.untracked.bool(True)                                                                    

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


# Select events based on the pixel cluster multiplicity
import  HLTrigger.special.hltPixelActivityFilter_cfi
process.multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
    inputTag  = cms.InputTag('siPixelClusters'),
    minClusters = cms.uint32(10000),
    maxClusters = cms.uint32(50000)
)

from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
process.PixelLayerTriplets.BPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
process.PixelLayerTriplets.FPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
process.pixelTracksHitTriplets.SeedComparitorPSet.clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting'

process.RecoForDQM_TrkReco = cms.Sequence(process.offlineBeamSpot*process.MeasurementTrackerEventPreSplitting*process.siPixelClusterShapeCachePreSplitting*process.recopixelvertexing*process.InitialStepPreSplitting)

process.p = cms.Path(
    process.scalersRawToDigi*
    process.APVPhases*
    process.consecutiveHEs*
    process.hltTriggerTypeFilter*
    process.siStripFEDCheck *
    process.RecoForDQM_LocalReco*
    process.siPixelClusters*
    process.multFilter*
    process.DQMCommon*
    process.SiStripClients*
    process.SiStripSources_LocalReco*
    ##### TRIGGER SELECTION #####
    process.hltHighLevel*
    process.RecoForDQM_TrkReco*
    process.SiStripSources_TrkReco*
    process.TrackingClient
    )
  
### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
