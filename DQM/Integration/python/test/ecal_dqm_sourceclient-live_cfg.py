### AUTO-GENERATED CMSRUN CONFIGURATION FOR ECAL DQM ###
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('runkey', default = 'pp_run', mult = VarParsing.multiplicity.singleton, mytype = VarParsing.varType.string, info = 'Run Keys of CMS')
options.register('runNumber', default = 194533, mult = VarParsing.multiplicity.singleton, mytype = VarParsing.varType.int, info = "Run number.")
options.register('runInputDir', default = '/fff/BU0/test', mult = VarParsing.multiplicity.singleton, mytype = VarParsing.varType.string, info = "Directory where the DQM files will appear.")
options.register('skipFirstLumis', default = False, mult = VarParsing.multiplicity.singleton, mytype = VarParsing.varType.bool, info = "Skip (and ignore the minEventsPerLumi parameter) for the files which have been available at the begining of the processing.")

options.parseArguments()


from DQM.Integration.test.dqmPythonTypes import *
runType = RunType(['pp_run','cosmic_run','hi_run','hpu_run'])
if not options.runkey.strip():
    options.runkey = 'pp_run'

runType.setRunType(options.runkey.strip())

process = cms.Process("process")

### Load cfis ###

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")
process.load("RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff")
process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi")
process.load("L1Trigger.Configuration.L1RawToDigi_cff")
process.load("DQM.EcalMonitorTasks.EcalMonitorTask_cfi")
process.load("DQM.EcalMonitorClient.EcalMonitorClient_cfi")
process.load("DQM.Integration.test.environment_cfi")
process.load("FWCore.Modules.preScaler_cfi")
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

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
    clusterPrescaleFactor = cms.untracked.int32(1),
    EcalRawDataCollection = cms.InputTag("ecalDigis")
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('WARNING'),
        noTimeStamps = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('cerr')
)

process.source = cms.Source("DQMStreamerReader",
    streamLabel = cms.untracked.string('_streamDQM_StorageManager'),
    delayMillis = cms.untracked.uint32(500),
    runNumber = cms.untracked.uint32(0),
    endOfRunKills = cms.untracked.bool(True),
    runInputDir = cms.untracked.string(''),
    minEventsPerLumi = cms.untracked.int32(1),
    deleteDatFiles = cms.untracked.bool(False),
    SelectEvents = cms.untracked.vstring('*'),
    skipFirstLumis = cms.untracked.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.simEcalTriggerPrimitiveDigis.InstanceEB = "ebDigis"
process.simEcalTriggerPrimitiveDigis.InstanceEE = "eeDigis"
process.simEcalTriggerPrimitiveDigis.Label = "ecalDigis"

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

process.ecalMonitorClient.verbosity = 0
process.ecalMonitorClient.workers = ['IntegrityClient', 'OccupancyClient', 'PresampleClient', 'RawDataClient', 'TimingClient', 'SelectiveReadoutClient', 'TrigPrimClient', 'SummaryClient']
process.ecalMonitorClient.workerParameters.SummaryClient.params.activeSources = ['Integrity', 'RawData', 'Presample', 'TriggerPrimitives', 'Timing', 'HotCell']
process.ecalMonitorClient.commonParameters.onlineMode = True

process.ecalMonitorTask.workers = ['ClusterTask', 'EnergyTask', 'IntegrityTask', 'OccupancyTask', 'RawDataTask', 'TimingTask', 'TrigPrimTask', 'PresampleTask', 'SelectiveReadoutTask']
process.ecalMonitorTask.verbosity = 0
process.ecalMonitorTask.collectionTags.EESuperCluster = "multi5x5SuperClusters:multi5x5EndcapSuperClusters"
process.ecalMonitorTask.collectionTags.EBBasicCluster = "hybridSuperClusters:hybridBarrelBasicClusters"
process.ecalMonitorTask.collectionTags.EEBasicCluster = "multi5x5SuperClusters:multi5x5EndcapBasicClusters"
process.ecalMonitorTask.collectionTags.Source = "rawDataCollector"
process.ecalMonitorTask.collectionTags.EBSuperCluster = "correctedHybridSuperClusters"
process.ecalMonitorTask.collectionTags.TrigPrimEmulDigi = "simEcalTriggerPrimitiveDigis"
process.ecalMonitorTask.workerParameters.TrigPrimTask.params.runOnEmul = True
process.ecalMonitorTask.commonParameters.willConvertToEDM = False
process.ecalMonitorTask.commonParameters.onlineMode = True

### Sequences ###

process.ecalPreRecoSequence = cms.Sequence(process.ecalDigis)
process.ecalRecoSequence = cms.Sequence((process.ecalMultiFitUncalibRecHit+process.ecalDetIdToBeRecovered+process.ecalRecHit)+(process.simEcalTriggerPrimitiveDigis+process.gtDigis)+(process.hybridClusteringSequence+process.multi5x5ClusteringSequence))
process.multi5x5ClusteringSequence = cms.Sequence(process.multi5x5BasicClustersCleaned+process.multi5x5SuperClustersCleaned+process.multi5x5BasicClustersUncleaned+process.multi5x5SuperClustersUncleaned+process.multi5x5SuperClusters)
process.hybridClusteringSequence = cms.Sequence(process.cleanedHybridSuperClusters+process.uncleanedHybridSuperClusters+process.hybridSuperClusters+process.correctedHybridSuperClusters+process.uncleanedOnlyCorrectedHybridSuperClusters)

### Paths ###

process.ecalMonitorPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalPhysicsFilter+process.ecalRecoSequence+process.ecalMonitorTask)
process.ecalClientPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalPhysicsFilter+process.ecalMonitorClient)

process.dqmEndPath = cms.EndPath(process.dqmEnv)
process.dqmOutputPath = cms.EndPath(process.dqmSaver)

### Schedule ###

process.schedule = cms.Schedule(process.ecalMonitorPath,process.ecalClientPath,process.dqmEndPath,process.dqmOutputPath)

### Setup source ###
process.source.runNumber = options.runNumber
process.source.runInputDir = options.runInputDir
process.source.skipFirstLumis = options.skipFirstLumis

### Run type specific ###

referenceFileName = process.DQMStore.referenceFileName.pythonValue()
if runType.getRunType() == runType.pp_run:
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_pp.root')
elif runType.getRunType() == runType.cosmic_run:
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_cosmic.root')
#    process.dqmEndPath.remove(process.dqmQTest)
    process.ecalMonitorTask.workers = ['EnergyTask', 'IntegrityTask', 'OccupancyTask', 'RawDataTask', 'TrigPrimTask', 'PresampleTask', 'SelectiveReadoutTask']
    process.ecalMonitorClient.workers = ['IntegrityClient', 'OccupancyClient', 'PresampleClient', 'RawDataClient', 'SelectiveReadoutClient', 'TrigPrimClient', 'SummaryClient']
    process.ecalMonitorClient.workerParameters.SummaryClient.params.activeSources = ['Integrity', 'RawData', 'Presample', 'TriggerPrimitives', 'HotCell']
elif runType.getRunType() == runType.hi_run:
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_hi.root')
elif runType.getRunType() == runType.hpu_run:
    process.DQMStore.referenceFileName = referenceFileName.replace('.root', '_hpu.root')
    process.source.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('*'))
