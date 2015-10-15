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
process.load("DQM.EcalMonitorTasks.EcalCalibMonitorTasks_cfi")
process.load("DQM.EcalMonitorTasks.EcalMonitorTask_cfi")
process.load("DQM.EcalMonitorClient.EcalCalibMonitorClient_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("FWCore.Modules.preScaler_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

### Individual module setups ###
from DQM.EcalCommon.CommonParams_cfi import ecaldqmLaserWavelengths, ecaldqmMGPAGains, ecaldqmMGPAGainsPN
ecaldqmMGPAGains.append(1)
ecaldqmMGPAGains.append(6)

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
    input = cms.untracked.int32(-1)
)

process.ecalLaserLedUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    algo = cms.string('EcalUncalibRecHitWorkerFixedAlphaBetaFit'),
    algoPSet = cms.PSet(
    	alphaEB = cms.double(1.138),
    	alphaEE = cms.double(1.89),
        AlphaBetaFilename = cms.untracked.string('NOFILE'),
    	betaEB = cms.double(1.655),
        betaEE = cms.double(1.4),
    	MinAmplEndcap = cms.double(16.0),
    	MinAmplBarrel = cms.double(12.0),
    	UseDynamicPedestal = cms.bool(True)
    )
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
    InputLabel = cms.InputTag("hltEcalCalibrationRaw"),
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
    fileNames = cms.untracked.vstring('/store/data/Run2015A/TestEnablesEcalHcal/RAW/v1/000/250/882/00000/0C59A756-CA22-E511-95A5-02163E012180.root'),
#                                      '/store/data/Run2015A/TestEnablesEcalHcal/RAW/v1/000/250/882/00000/40560E96-CF22-E511-B7CE-02163E0146D1.root')
    skipEvents=cms.untracked.uint32(100000)
)

process.DQM = cms.Service("DQM",
    filter = cms.untracked.string(''),
    publishFrequency = cms.untracked.double(5.0),
    collectorHost = cms.untracked.string(''),
    collectorPort = cms.untracked.int32(0),
    debug = cms.untracked.bool(False)
)

process.ecalTestPulseUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    algo = cms.string('EcalUncalibRecHitWorkerMaxSample'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)

process.ecalCalibMonitorClient.verbosity = 0
process.ecalCalibMonitorClient.workers = ['IntegrityClient', 'RawDataClient', 'PedestalClient', 'TestPulseClient', 'LaserClient', 'LedClient', 'PNIntegrityClient', 'SummaryClient', 'CalibrationSummaryClient']
process.ecalCalibMonitorClient.workerParameters.SummaryClient.params.activeSources = ['Integrity', 'RawData']

process.preScaler.prescaleFactor = 1

process.ecalPedestalMonitorTask.verbosity = 0

process.dqmSaver.convention = "Offline"
process.dqmSaver.referenceHandling = "skip"
process.dqmSaver.version = 2
process.dqmSaver.workflow = "/All/Run2015/CentralDAQ"

process.ecalMonitorTask.workers = ['IntegrityTask', 'RawDataTask']
process.ecalMonitorTask.collectionTags.Source = "hltEcalCalibrationRaw"

process.ecalLaserLedMonitorTask.verbosity = 0
process.ecalLaserLedMonitorTask.collectionTags.EBLaserLedUncalibRecHit = "ecalLaserLedUncalibRecHit:EcalUncalibRecHitsEB"
process.ecalLaserLedMonitorTask.collectionTags.EELaserLedUncalibRecHit = "ecalLaserLedUncalibRecHit:EcalUncalibRecHitsEE"

process.ecalTestPulseMonitorTask.verbosity = 0

process.ecalRecHit.EEuncalibRecHitCollection = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEE"
process.ecalRecHit.EBuncalibRecHitCollection = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEB"

process.ecalPNDiodeMonitorTask.verbosity = 0

process.dqmEnv.subSystemFolder = cms.untracked.string('EcalCalibration')

### Sequences ###

process.ecalRecoSequence = cms.Sequence((process.ecalGlobalUncalibRecHit+process.ecalDetIdToBeRecovered+process.ecalRecHit))
process.ecalPreRecoSequence = cms.Sequence(process.ecalDigis)

### Paths ###

process.ecalLaserLedPath = cms.Path(process.ecalPreRecoSequence+process.ecalRecoSequence+process.ecalLaserLedUncalibRecHit+process.ecalLaserLedMonitorTask+process.ecalPNDiodeMonitorTask)
process.ecalTestPulsePath = cms.Path(process.ecalPreRecoSequence+process.ecalRecoSequence+process.ecalTestPulseUncalibRecHit+process.ecalTestPulseMonitorTask+process.ecalPNDiodeMonitorTask)
process.ecalPedestalPath = cms.Path(process.ecalPreRecoSequence+process.ecalRecoSequence+process.ecalPedestalMonitorTask+process.ecalPNDiodeMonitorTask)
process.ecalMonitorPath = cms.Path(process.ecalPreRecoSequence+process.ecalMonitorTask)
process.ecalClientPath = cms.Path(process.ecalCalibMonitorClient)

process.dqmEndPath = cms.EndPath(process.dqmEnv)
process.dqmOutputPath = cms.EndPath(process.dqmSaver)

### Schedule ###

process.schedule = cms.Schedule(process.ecalLaserLedPath,process.ecalTestPulsePath,process.ecalPedestalPath,process.ecalMonitorPath,process.ecalClientPath,process.dqmEndPath,process.dqmOutputPath)

### Setup source ###

if options.inputFiles:
    process.source.fileNames = options.inputFiles
if options.maxEvents != -1:
    process.maxEvents.input = options.maxEvents
