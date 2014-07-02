import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStrpDQMLive")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siStripDigis', 
                                         'siStripClusters', 
                                         'siStripZeroSuppression', 
                                        'SiStripClusterizer'),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
    destinations = cms.untracked.vstring('cout')
)

#----------------------------
# Event Source
#-----------------------------
# for live online DQM in P5
#process.load("DQM.Integration.test.inputsource_cfi")

# for testing in lxplus
process.load("DQM.Integration.test.fileinputsource_cfi")

#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
# DQM Live Environment
#-----------------------------
#from DQM.Integration.test.environment_cfi import HEAVYION

process.load("DQM.Integration.test.environment_cfi")
process.DQM.filter = '^(SiStrip|Tracking)(/[^/]+){0,5}$'

process.dqmEnv.subSystemFolder    = "SiStrip"
process.dqmSaver.producer = "Playback"
process.dqmSaver.saveByTime = 60
process.dqmSaver.saveByMinute = 60

# uncomment for running in local
process.dqmSaver.dirName     = '.'

process.dqmEnvTr = cms.EDAnalyzer("DQMEventInfo",
                 subSystemFolder = cms.untracked.string('Tracking'),
                 eventRateWindow = cms.untracked.double(0.5),
                 eventInfoFolder = cms.untracked.string('EventInfo')
)

## uncooment for running in local
## collector
#process.DQM.collectorHost = 'vmepcs2b18-20.cms'
#process.DQM.collectorPort = 9190

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
process.load("Configuration.StandardSequences.Geometry_cff")

#--------------------------
# Calibration
#--------------------------
# Condition for P5 cluster
#process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
# Condition for lxplus
process.load("DQM.Integration.test.FrontierCondition_GT_Offline_cfi") 

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
process.load("Configuration.StandardSequences.Reconstruction_cff")

## Cosmic Track Reconstruction
if (process.runType.getRunType() == process.runType.cosmic_run):
    process.load("RecoTracker.Configuration.RecoTrackerP5_cff")

## # offline beam spot
## process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
## # online beam spot
## process.load("RecoVertex.BeamSpotProducer.BeamSpotOnline_cff")

# handle onlineBeamSpot w/o changing all configuration
# the same shortcut is also used in Express ;)
# http://cmslxr.fnal.gov/lxr/source/Configuration/DataProcessing/python/RecoTLR.py#044
import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

#--------------------------
# Strip DQM Source and Client
#--------------------------
process.load("DQM.SiStripMonitorClient.SiStripSourceConfigP5_cff")
process.TrackMon_gentk.doLumiAnalysis = False
process.TrackMon_ckf.doLumiAnalysis = False
process.TrackMon_hi.doLumiAnalysis = False
process.TrackMon_ckf.AlgoName = 'CKFTk'

#--------------------------
# Quality Test
#--------------------------
process.qTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
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
                   EventsToSkip = cms.untracked.int32(3)
)

#--------------------------
# Producers
#--------------------------
# Event History Producer
process.load("DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi")

# APV Phase Producer
#process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2011_cfi")
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
process.hltHighLevel.HLTPaths = cms.vstring( 'HLT_ZeroBias_*' , 'HLT_ZeroBias1_*' , 'HLT_PAZeroBias_*' , 'HLT_PAZeroBias1_*' )
process.hltHighLevel.andOr = cms.bool(True)
process.hltHighLevel.throw =  cms.bool(False)

#--------------------------
# Scheduling
#--------------------------
process.SiStripSources_LocalReco = cms.Sequence(process.siStripFEDMonitor*process.SiStripMonitorDigi*process.SiStripMonitorClusterReal)
process.DQMCommon                = cms.Sequence(process.qTester*process.dqmEnv*process.dqmEnvTr*process.dqmSaver)
process.RecoForDQM_LocalReco     = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.gtDigis*process.trackerlocalreco*process.gtEvmDigis)

#--------------------------
# Global Plot Switches
#--------------------------
process.SiStripMonitorClusterReal.TH1ClusterCharge.moduleswitchon = True


if (process.runType.getRunType() == process.runType.cosmic_run):
    # event selection for cosmic data
#    process.DQMStreamHttpReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('HLT_L1*','HLT_*Cosmic*','HLT_ZeroBias*'))
    # Reference run for cosmic
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/sistrip_reference_cosmic.root'
    # Source and Client config for cosmic data
    process.SiStripSources_TrkReco_cosmic = cms.Sequence(process.SiStripMonitorTrack_ckf*process.TrackMon_ckf)
    process.load("DQM.SiStripMonitorClient.SiStripClientConfigP5_Cosmic_cff")
    process.SiStripAnalyserCosmic.RawDataTag = cms.untracked.InputTag("rawDataCollector")
    process.SiStripAnalyserCosmic.TkMapCreationFrequency  = -1
    process.SiStripAnalyserCosmic.ShiftReportFrequency = -1
    process.SiStripAnalyserCosmic.StaticUpdateFrequency = 5
    process.SiStripClients           = cms.Sequence(process.SiStripAnalyserCosmic)

    # Reco for cosmic data
    process.load('RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cfi')
    process.simpleCosmicBONSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 450

    process.RecoForDQM_TrkReco_cosmic = cms.Sequence(process.offlineBeamSpot*process.ctftracksP5)

    process.qTester = cms.EDAnalyzer("QualityTester",
                                     qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_cosmic.xml'),
                                     prescaleFactor = cms.untracked.int32(2),
                                     getQualityTestsFromFile = cms.untracked.bool(True),
                                     qtestOnEndLumi = cms.untracked.bool(True),
                                     qtestOnEndRun = cms.untracked.bool(True)
                                     )

    process.p = cms.Path(process.scalersRawToDigi*
                         process.APVPhases*
                         process.consecutiveHEs*
                         process.hltTriggerTypeFilter*
                         process.RecoForDQM_LocalReco*
                         process.DQMCommon*
                         process.SiStripClients*
                         process.SiStripSources_LocalReco*
                         process.RecoForDQM_TrkReco_cosmic*
                         process.SiStripSources_TrkReco_cosmic
                         )



#else :
if (process.runType.getRunType() == process.runType.pp_run):
    #event selection for pp collisions
#    process.DQMEventStreamHttpReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('HLT_L1*',
#                                                                                                  'HLT_Jet*',
#                                                                                                  'HLT_HT*',
#                                                                                                  'HLT_MinBias_#*',
#                                                                                                  'HLT_Physics*#',
#                                                                                                  'HLT_ZeroBias#*',
#                                                                                                  'HLT_PAL1*',
#                                                                                                  'HLT_PAZeroBias_*'))    
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/sistrip_reference_pp.root'
    # Source and Client config for pp collisions

    process.SiStripMonitorDigi.UseDCSFiltering = cms.bool(False)
    process.SiStripMonitorClusterReal.UseDCSFiltering = cms.bool(False)
    
    process.SiStripSources_TrkReco   = cms.Sequence(process.SiStripMonitorTrack_gentk*process.MonitorTrackResiduals_gentk*process.TrackMon_gentk)
    process.load("DQM.SiStripMonitorClient.SiStripClientConfigP5_cff")
    process.SiStripAnalyser.UseGoodTracks  = cms.untracked.bool(True)
    process.SiStripAnalyser.TkMapCreationFrequency  = -1
    process.SiStripAnalyser.ShiftReportFrequency = -1
    process.SiStripAnalyser.StaticUpdateFrequency = 5
    process.SiStripAnalyser.RawDataTag = cms.untracked.InputTag("rawDataCollector")
    process.SiStripClients           = cms.Sequence(process.SiStripAnalyser)

    process.SiStripMonitorDigi.TotalNumberOfDigisFailure.integrateNLumisections = cms.int32(25)
    
    # Reco for pp collisions

    process.load('RecoTracker.Configuration.RecoTracker_cff')
    
    process.newCombinedSeeds.seedCollections = cms.VInputTag(
        cms.InputTag('initialStepSeeds'),
        )
    
    process.load('RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff')
    process.generalTracks.TrackProducers = (
        cms.InputTag('initialStepTracks'),
        )
            
    process.generalTracks.hasSelector=cms.vint32(1)
    process.generalTracks.selectedTrackQuals = cms.VInputTag(
        cms.InputTag("initialStepSelector","initialStep"),
        )
    process.generalTracks.setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0), pQual=cms.bool(True) ) )

    process.load("RecoTracker.IterativeTracking.iterativeTk_cff")

    process.iterTracking_FirstStep =cms.Sequence(
        process.InitialStep
        *process.generalTracks
        )

    #process.RecoForDQM_TrkReco       = cms.Sequence(process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks)
    process.RecoForDQM_TrkReco       = cms.Sequence(process.offlineBeamSpot*process.recopixelvertexing*process.iterTracking_FirstStep)

    process.p = cms.Path(process.scalersRawToDigi*
                         process.APVPhases*
                         process.consecutiveHEs*
                         process.hltTriggerTypeFilter*
                         process.RecoForDQM_LocalReco*
                         process.DQMCommon*
                         process.SiStripClients*
                         process.SiStripSources_LocalReco*
                         process.hltHighLevel *
                         process.RecoForDQM_TrkReco*
                         process.SiStripSources_TrkReco
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
    process.eventFilter.EventsToSkip = cms.untracked.int32(30)
    
    # change the HLT trigger path selection
    # it should already be ok, but the name could be changed
    process.hltHighLevel.HLTPaths = cms.vstring( 'HLT_ZeroBias*' )

#    process.DQMEventStreamHttpReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('HLT_600Tower*','HLT_L1*','HLT_Jet*','HLT_HT*','HLT_MinBias_*','HLT_Physics*', 'HLT_ZeroBias*'))
#
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/sistrip_reference_pp.root'

    process.SiStripMonitorDigi.UseDCSFiltering = cms.bool(False)
    process.SiStripMonitorClusterReal.UseDCSFiltering = cms.bool(False)
    
    process.SiStripSources_TrkReco   = cms.Sequence(process.SiStripMonitorTrack_gentk*process.MonitorTrackResiduals_gentk*process.TrackMon_gentk)
    process.load("DQM.SiStripMonitorClient.SiStripClientConfigP5_cff")
    process.SiStripAnalyser.UseGoodTracks  = cms.untracked.bool(True)
    process.SiStripAnalyser.TkMapCreationFrequency  = -1
    process.SiStripAnalyser.ShiftReportFrequency = -1
    process.SiStripAnalyser.StaticUpdateFrequency = 5
    process.SiStripAnalyser.RawDataTag = cms.untracked.InputTag("rawDataCollector")
    process.SiStripClients           = cms.Sequence(process.SiStripAnalyser)

    # Reco for pp collisions

    process.load('RecoTracker.Configuration.RecoTracker_cff')
    
    process.newCombinedSeeds.seedCollections = cms.VInputTag(
        cms.InputTag('initialStepSeeds'),
        )
    
    process.load('RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff')
    process.generalTracks.TrackProducers = (
        cms.InputTag('initialStepTracks'),
        )
            
    process.generalTracks.hasSelector=cms.vint32(1)
    process.generalTracks.selectedTrackQuals = cms.VInputTag(
        cms.InputTag("initialStepSelector","initialStep"),
        )
    process.generalTracks.setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0), pQual=cms.bool(True) ) )

    process.load("RecoTracker.IterativeTracking.iterativeTk_cff")

    process.iterTracking_FirstStep =cms.Sequence(
        process.InitialStep
        *process.generalTracks
        )

    #process.RecoForDQM_TrkReco       = cms.Sequence(process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks)
    process.RecoForDQM_TrkReco       = cms.Sequence(process.offlineBeamSpot*process.recopixelvertexing*process.iterTracking_FirstStep)

    process.p = cms.Path(process.scalersRawToDigi*
                         process.APVPhases*
                         process.consecutiveHEs*
                         process.hltTriggerTypeFilter*
                         process.RecoForDQM_LocalReco*
                         process.DQMCommon*
                         process.SiStripClients*
                         process.SiStripSources_LocalReco*
                         process.hltHighLevel *
                         process.eventFilter*
                         process.RecoForDQM_TrkReco*
                         process.SiStripSources_TrkReco
                         )

process.castorDigis.InputLabel = cms.InputTag("rawDataCollector")
process.csctfDigis.producer = cms.InputTag("rawDataCollector")
process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataCollector")
process.ecalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataCollector")
process.gctDigis.inputLabel = cms.InputTag("rawDataCollector")
process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataCollector")
process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
process.muonDTDigis.inputLabel = cms.InputTag("rawDataCollector")
process.muonRPCDigis.InputLabel = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")
process.siStripFEDMonitor.RawDataTag = cms.untracked.InputTag("rawDataCollector")
#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print "Running with run type = ", process.runType.getRunType()

if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
    process.siStripFEDMonitor.RawDataTag = cms.untracked.InputTag("rawDataRepacker")

    # Select events based on the pixel cluster multiplicity
    import  HLTrigger.special.hltPixelActivityFilter_cfi
    process.multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
        inputTag  = cms.InputTag('siPixelClusters'),
        minClusters = cms.uint32(10000),
        maxClusters = cms.uint32(50000)
        )
    # Trigger selection
#    process.DQMEventStreamHttpReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('HLT_HIJet*','HLT_HICentralityVeto*','HLT_HIFullTrack*','HLT_HIMinBias*'))
#
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/sistrip_reference_hi.root'
    # Quality test for HI                                                                                                                  
    process.qTester = cms.EDAnalyzer("QualityTester",
                                     qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_heavyion.xml'),
                                     prescaleFactor = cms.untracked.int32(3),
                                     getQualityTestsFromFile = cms.untracked.bool(True),
                                     qtestOnEndLumi = cms.untracked.bool(True),
                                     qtestOnEndRun = cms.untracked.bool(True)
                                     )

    # Sources for HI 
    process.load("Configuration.StandardSequences.RawToDigi_Repacked_cff")
    process.SiStripBaselineValidator.srcProcessedRawDigi =  cms.InputTag('siStripVRDigis','VirginRaw')
    process.SiStripSources_TrkReco   = cms.Sequence(process.SiStripMonitorTrack_hi*process.TrackMon_hi)
# Client for HI
    process.load("DQM.SiStripMonitorClient.SiStripClientConfigP5_HeavyIons_cff")
    process.SiStripAnalyserHI.RawDataTag = cms.untracked.InputTag("rawDataRepacker")
    process.SiStripAnalyserHI.TkMapCreationFrequency  = -1
    process.SiStripAnalyserHI.ShiftReportFrequency = -1
    process.SiStripAnalyserHI.StaticUpdateFrequency = 5
    process.SiStripClients  = cms.Sequence(process.SiStripAnalyserHI)
    # Reco for HI collisions
    process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff")
    process.RecoForDQM_LocalReco = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.siStripVRDigis*process.gtDigis*process.trackerlocalreco)
    process.RecoForDQM_TrkReco = cms.Sequence(process.offlineBeamSpot*process.heavyIonTracking)
    
    process.p = cms.Path(process.scalersRawToDigi*
                         process.APVPhases*
                         process.consecutiveHEs*
                         process.hltTriggerTypeFilter*
                         process.RecoForDQM_LocalReco*
                         process.DQMCommon*
                         process.SiStripClients*
                         process.SiStripSources_LocalReco*
                         process.RecoForDQM_TrkReco*
                         process.SiStripSources_TrkReco*
                         process.multFilter*
                         process.SiStripBaselineValidator                         
                         )

