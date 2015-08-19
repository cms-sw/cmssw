### AUTO-GENERATED CMSRUN CONFIGURATION FOR ECAL DQM ###
import FWCore.ParameterSet.Config as cms

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
process.load("DQM.EcalMonitorTasks.EcalMonitorTask_cfi")
process.load("DQM.EcalMonitorClient.EcalMonitorClient_cfi")
process.load("DQM.Integration.config.environment_cfi")
process.load("FWCore.Modules.preScaler_cfi")
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
process.load("DQM.Integration.config.inputsource_cfi")

### Individual module setups ###

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

process.ecalPhysicsFilter = cms.EDFilter("EcalMonitorPrescaler",
    cosmics = cms.untracked.uint32(1),
    physics = cms.untracked.uint32(1),
    EcalRawDataCollection = cms.InputTag("ecalDigis")
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        EcalLaserDbService = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        ),
        noTimeStamps = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalDQM = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('INFO')
    ),
    categories = cms.untracked.vstring('EcalDQM', 
        'EcalLaserDbService'),
    destinations = cms.untracked.vstring('cerr', 
        'cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.ecalMonitorClient.verbosity = 0
process.ecalMonitorClient.workers = ['IntegrityClient', 'OccupancyClient', 'PresampleClient', 'RawDataClient', 'TimingClient', 'SelectiveReadoutClient', 'TrigPrimClient', 'SummaryClient']
process.ecalMonitorClient.workerParameters.SummaryClient.params.activeSources = ['Integrity', 'RawData', 'Presample', 'TriggerPrimitives', 'Timing', 'HotCell']
process.ecalMonitorClient.commonParameters.onlineMode = True

process.GlobalTag.toGet = cms.VPSet(cms.PSet(
    record = cms.string('EcalDQMChannelStatusRcd'),
    tag = cms.string('EcalDQMChannelStatus_v1_hlt'),
    connect = cms.untracked.string('frontier://(proxyurl=http://frontier.cms:3128)(serverurl=http://frontier.cms:8000/FrontierOnProd)(serverurl=http://frontier.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_34X_ECAL')
), 
    cms.PSet(
        record = cms.string('EcalDQMTowerStatusRcd'),
        tag = cms.string('EcalDQMTowerStatus_v1_hlt'),
        connect = cms.untracked.string('frontier://(proxyurl=http://frontier.cms:3128)(serverurl=http://frontier.cms:8000/FrontierOnProd)(serverurl=http://frontier.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_34X_ECAL')
    ))

process.preScaler.prescaleFactor = 1

process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/ecal_reference.root"

process.dqmEnv.subSystemFolder = cms.untracked.string('Ecal')
process.dqmSaver.tag = cms.untracked.string('Ecal')

process.simEcalTriggerPrimitiveDigis.InstanceEB = "ebDigis"
process.simEcalTriggerPrimitiveDigis.InstanceEE = "eeDigis"
process.simEcalTriggerPrimitiveDigis.Label = "ecalDigis"

process.ecalRecHit.EEuncalibRecHitCollection = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEE"
process.ecalRecHit.EBuncalibRecHitCollection = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEB"

process.ecalMonitorTask.workers = ['ClusterTask', 'EnergyTask', 'IntegrityTask', 'OccupancyTask', 'RawDataTask', 'TimingTask', 'TrigPrimTask', 'PresampleTask', 'SelectiveReadoutTask']
process.ecalMonitorTask.verbosity = 0
process.ecalMonitorTask.collectionTags.EEUncalibRecHit = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEE"
process.ecalMonitorTask.collectionTags.EESuperCluster = "multi5x5SuperClusters:multi5x5EndcapSuperClusters"
process.ecalMonitorTask.collectionTags.EBBasicCluster = "hybridSuperClusters:hybridBarrelBasicClusters"
process.ecalMonitorTask.collectionTags.EEBasicCluster = "multi5x5SuperClusters:multi5x5EndcapBasicClusters"
process.ecalMonitorTask.collectionTags.Source = "rawDataCollector"
process.ecalMonitorTask.collectionTags.EBSuperCluster = "correctedHybridSuperClusters"
process.ecalMonitorTask.collectionTags.TrigPrimEmulDigi = "simEcalTriggerPrimitiveDigis"
process.ecalMonitorTask.collectionTags.EBUncalibRecHit = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEB"
process.ecalMonitorTask.workerParameters.TrigPrimTask.params.runOnEmul = True
process.ecalMonitorTask.commonParameters.willConvertToEDM = False
process.ecalMonitorTask.commonParameters.onlineMode = True

### Sequences ###

process.ecalPreRecoSequence = cms.Sequence(process.ecalDigis)
process.ecalRecoSequence = cms.Sequence((process.ecalGlobalUncalibRecHit+process.ecalDetIdToBeRecovered+process.ecalRecHit)+(process.simEcalTriggerPrimitiveDigis+process.gtDigis)+(process.hybridClusteringSequence+process.multi5x5ClusteringSequence))
process.multi5x5ClusteringSequence = cms.Sequence(process.multi5x5BasicClustersCleaned+process.multi5x5SuperClustersCleaned+process.multi5x5BasicClustersUncleaned+process.multi5x5SuperClustersUncleaned+process.multi5x5SuperClusters)
process.hybridClusteringSequence = cms.Sequence(process.cleanedHybridSuperClusters+process.uncleanedHybridSuperClusters+process.hybridSuperClusters+process.correctedHybridSuperClusters+process.uncleanedOnlyCorrectedHybridSuperClusters)

### Paths ###

process.ecalMonitorPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalPhysicsFilter+process.ecalRecoSequence+process.ecalMonitorTask)
process.ecalClientPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalPhysicsFilter+process.ecalMonitorClient)

process.dqmEndPath = cms.EndPath(process.dqmEnv)
process.dqmOutputPath = cms.EndPath(process.dqmSaver)

### Schedule ###

process.schedule = cms.Schedule(process.ecalMonitorPath,process.ecalClientPath,process.dqmEndPath,process.dqmOutputPath)

### Run type specific ###

referenceFileName = process.DQMStore.referenceFileName.pythonValue()
runTypeName = process.runType.getRunTypeName()
if (runTypeName == 'pp_run' or runTypeName == 'pp_run_stage1'):
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_pp.root')
elif (runTypeName == 'cosmic_run' or runTypeName == 'cosmic_run_stage1'):
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_cosmic.root')
#    process.dqmEndPath.remove(process.dqmQTest)
    process.ecalMonitorTask.workers = ['EnergyTask', 'IntegrityTask', 'OccupancyTask', 'RawDataTask', 'TimingTask', 'TrigPrimTask', 'PresampleTask', 'SelectiveReadoutTask']
    process.ecalMonitorClient.workers = ['IntegrityClient', 'OccupancyClient', 'PresampleClient', 'RawDataClient', 'TimingClient', 'SelectiveReadoutClient', 'TrigPrimClient', 'SummaryClient']
    process.ecalMonitorClient.workerParameters.SummaryClient.params.activeSources = ['Integrity', 'RawData', 'Presample', 'TriggerPrimitives', 'Timing', 'HotCell']
elif runTypeName == runType.hi_run:
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_hi.root')
elif runTypeName == runType.hpu_run:
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_hpu.root')
    process.source.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('*'))


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
