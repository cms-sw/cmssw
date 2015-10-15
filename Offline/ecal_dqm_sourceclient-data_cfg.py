### AUTO-GENERATED CMSRUN CONFIGURATION FOR ECAL DQM ###
import FWCore.ParameterSet.Config as cms


from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.parseArguments()

process = cms.Process("process")

### Load cfis ###

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")
process.load("RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff")
process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi")
process.load("L1Trigger.Configuration.L1RawToDigi_cff")
process.load("RecoEcal.EgammaCoreTools.EcalNextToDeadChannelESProducer_cff")
process.load("RecoEcal.EgammaClusterProducers.reducedRecHitsSequence_cff")
process.load("DQM.EcalMonitorTasks.EcalMonitorTask_cfi")
process.load("DQM.EcalMonitorClient.EcalMonitorClient_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("FWCore.Modules.preScaler_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

### Individual module setups ###

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('WARNING'),
        noTimeStamps = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalLaserDbService = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        ),
        threshold = cms.untracked.string('INFO'),
        EcalDQM = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('EcalDQM', 
        'EcalLaserDbService'),
    destinations = cms.untracked.vstring('cerr', 
        'cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.GlobalTag = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(60),
        authenticationSystem = cms.untracked.int32(0),
        connectionRetrialPeriod = cms.untracked.int32(10)
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalDQMChannelStatusRcd'),
        tag = cms.string('EcalDQMChannelStatus_v1_hlt'),
        connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
    ),
        cms.PSet(
            record = cms.string('EcalDQMTowerStatusRcd'),
            tag = cms.string('EcalDQMTowerStatus_v1_hlt'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        )),
    connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
    globaltag = cms.string('GR_H_V58C')
)

process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')


process.ecalDigis = cms.EDProducer("EcalRawToDigi",
    tccUnpacking = cms.bool(True),
    FedLabel = cms.InputTag("listfeds"),
    srpUnpacking = cms.bool(True),
    syncCheck = cms.bool(True),
    feIdCheck = cms.bool(True),
    silentMode = cms.untracked.bool(True),
    InputLabel = cms.InputTag("rawDataCollector"),
    orderedFedList = cms.vint32(601, 602, 603, 604, 605, 
        606, 607, 608, 609, 610, 
        611, 612, 613, 614, 615, 
        616, 617, 618, 619, 620, 
        621, 622, 623, 624, 625, 
        626, 627, 628, 629, 630, 
        631, 632, 633, 634, 635, 
        636, 637, 638, 639, 640, 
        641, 642, 643, 644, 645, 
        646, 647, 648, 649, 650, 
        651, 652, 653, 654),
    eventPut = cms.bool(True),
    numbTriggerTSamples = cms.int32(1),
    numbXtalTSamples = cms.int32(10),
    orderedDCCIdList = cms.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25, 
        26, 27, 28, 29, 30, 
        31, 32, 33, 34, 35, 
        36, 37, 38, 39, 40, 
        41, 42, 43, 44, 45, 
        46, 47, 48, 49, 50, 
        51, 52, 53, 54),
    FEDs = cms.vint32(601, 602, 603, 604, 605, 
        606, 607, 608, 609, 610, 
        611, 612, 613, 614, 615, 
        616, 617, 618, 619, 620, 
        621, 622, 623, 624, 625, 
        626, 627, 628, 629, 630, 
        631, 632, 633, 634, 635, 
        636, 637, 638, 639, 640, 
        641, 642, 643, 644, 645, 
        646, 647, 648, 649, 650, 
        651, 652, 653, 654),
    DoRegional = cms.bool(False),
    feUnpacking = cms.bool(True),
    forceToKeepFRData = cms.bool(False),
    headerUnpacking = cms.bool(True),
    memUnpacking = cms.bool(True)
)

process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/data/Commissioning2015/MinimumBias/RAW/v1/000/232/881/00000/E63DF8F8-3FAB-E411-A41A-02163E011DD5.root")
    fileNames = cms.untracked.vstring('/store/data/Run2015A/MinimumBias/RAW/v1/000/246/993/00000/88A02D25-A20A-E511-B6BA-02163E0138C6.root')
)

process.DQM = cms.Service("DQM",
    filter = cms.untracked.string(''),
    publishFrequency = cms.untracked.double(5.0),
    collectorHost = cms.untracked.string(''),
    collectorPort = cms.untracked.int32(0),
    debug = cms.untracked.bool(False)
)

process.ecalMonitorTask.verbosity = 0
process.ecalMonitorClient.verbosity = 0

process.preScaler.prescaleFactor = 1

process.dqmEnv.subSystemFolder = cms.untracked.string('Ecal')

process.dqmSaver.convention = "Offline"
process.dqmSaver.referenceHandling = "skip"
process.dqmSaver.version = 1
process.dqmSaver.workflow = "/All/Run2015/CentralDAQ"

process.simEcalTriggerPrimitiveDigis.InstanceEB = "ebDigis"
process.simEcalTriggerPrimitiveDigis.InstanceEE = "eeDigis"
process.simEcalTriggerPrimitiveDigis.Label = "ecalDigis"

process.reducedEcalRecHitsEE.interestingDetIdCollections = [cms.InputTag("interestingEcalDetIdEE")]

process.ecalRecHit.EEuncalibRecHitCollection = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEE"
process.ecalRecHit.EBuncalibRecHitCollection = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEB"

process.ecalMonitorTask.collectionTags.EEUncalibRecHit = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEE"
process.ecalMonitorTask.collectionTags.EBBasicCluster = "hybridSuperClusters:hybridBarrelBasicClusters"
process.ecalMonitorTask.collectionTags.EEBasicCluster = "multi5x5SuperClusters:multi5x5EndcapBasicClusters"
process.ecalMonitorTask.collectionTags.EESuperCluster = "multi5x5SuperClusters:multi5x5EndcapSuperClusters"
process.ecalMonitorTask.collectionTags.EBSuperCluster = "correctedHybridSuperClusters"
process.ecalMonitorTask.collectionTags.TrigPrimEmulDigi = "simEcalTriggerPrimitiveDigis"
process.ecalMonitorTask.collectionTags.EBUncalibRecHit = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEB"
process.ecalMonitorTask.commonParameters.willConvertToEDM = False

process.reducedEcalRecHitsEB.interestingDetIdCollections = [cms.InputTag("interestingEcalDetIdEB")]

### Sequences ###

#process.ecalPreRecoSequence = cms.Sequence(process.ecalDigis+process.splashFilter)
process.ecalPreRecoSequence = cms.Sequence(process.ecalDigis)
process.ecalRecoSequence = cms.Sequence((process.ecalGlobalUncalibRecHit+process.ecalDetIdToBeRecovered+process.ecalRecHit)+(process.simEcalTriggerPrimitiveDigis+process.gtDigis)+(process.hybridClusteringSequence+process.multi5x5ClusteringSequence)+(process.interestingEcalDetIdEB+process.interestingEcalDetIdEE+process.reducedEcalRecHitsEB+process.reducedEcalRecHitsEE))
process.multi5x5ClusteringSequence = cms.Sequence(process.multi5x5BasicClustersCleaned+process.multi5x5SuperClustersCleaned+process.multi5x5BasicClustersUncleaned+process.multi5x5SuperClustersUncleaned+process.multi5x5SuperClusters)
process.hybridClusteringSequence = cms.Sequence(process.cleanedHybridSuperClusters+process.uncleanedHybridSuperClusters+process.hybridSuperClusters+process.correctedHybridSuperClusters+process.uncleanedOnlyCorrectedHybridSuperClusters)

### Paths ###

process.ecalMonitorPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalRecoSequence+process.ecalMonitorTask)
process.ecalClientPath = cms.Path(process.ecalPreRecoSequence+process.ecalMonitorClient)

process.dqmEndPath = cms.EndPath(process.dqmEnv)
process.dqmOutputPath = cms.EndPath(process.dqmSaver)

### Schedule ###

process.schedule = cms.Schedule(process.ecalMonitorPath,process.ecalClientPath,process.dqmEndPath,process.dqmOutputPath)

### Setup source ###

if options.inputFiles:
    process.source.fileNames = options.inputFiles
if options.maxEvents != -1:
    process.maxEvents.input = options.maxEvents
