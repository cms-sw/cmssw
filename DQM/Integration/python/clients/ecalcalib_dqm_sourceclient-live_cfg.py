### AUTO-GENERATED CMSRUN CONFIGURATION FOR ECAL DQM ###
import FWCore.ParameterSet.Config as cms

process = cms.Process("process")

### Load cfis ###

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
#process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
#process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
#process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")
process.load("DQM.EcalMonitorTasks.EcalCalibMonitorTasks_cfi")
process.load("DQM.EcalMonitorClient.EcalCalibMonitorClient_cfi")
process.load("DQM.Integration.config.environment_cfi")
process.load("FWCore.Modules.preScaler_cfi")
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
process.load("DQM.Integration.config.inputsource_cfi")
from DQM.Integration.config.inputsource_cfi import options

### Individual module setups ###

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

process.ecalLaserLedFilter = cms.EDFilter("EcalMonitorPrescaler",
    laser = cms.untracked.uint32(1),
    led = cms.untracked.uint32(1),
    EcalRawDataCollection = cms.InputTag("ecalDigis")
)

process.ecalPedestalFilter = cms.EDFilter("EcalMonitorPrescaler",
    pedestal = cms.untracked.uint32(1),
    EcalRawDataCollection = cms.InputTag("ecalDigis")
)

process.ecalTestPulseFilter = cms.EDFilter("EcalMonitorPrescaler",
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    testPulse = cms.untracked.uint32(1)
)

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

process.ecalTestPulseUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    algo = cms.string('EcalUncalibRecHitWorkerMaxSample'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)

process.ecalCalibMonitorClient.verbosity = 0
process.ecalCalibMonitorClient.commonParameters.onlineMode = True

process.preScaler.prescaleFactor = 1

process.source.streamLabel = "streamDQMCalibration"


process.ecalPedestalMonitorTask.verbosity = 0
process.ecalPedestalMonitorTask.commonParameters.onlineMode = True


process.ecalLaserLedMonitorTask.verbosity = 0
process.ecalLaserLedMonitorTask.collectionTags.EBLaserLedUncalibRecHit = "ecalLaserLedUncalibRecHit:EcalUncalibRecHitsEB"
process.ecalLaserLedMonitorTask.collectionTags.EELaserLedUncalibRecHit = "ecalLaserLedUncalibRecHit:EcalUncalibRecHitsEE"
process.ecalLaserLedMonitorTask.commonParameters.onlineMode = True

process.GlobalTag.toGet = cms.VPSet(cms.PSet(
    record = cms.string('EcalDQMChannelStatusRcd'),
    tag = cms.string('EcalDQMChannelStatus_v1_hlt'),
), 
    cms.PSet(
        record = cms.string('EcalDQMTowerStatusRcd'),
        tag = cms.string('EcalDQMTowerStatus_v1_hlt'),
    ))

process.ecalTestPulseMonitorTask.verbosity = 0
process.ecalTestPulseMonitorTask.commonParameters.onlineMode = True

process.ecalRecHit.EEuncalibRecHitCollection = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEE"
process.ecalRecHit.EBuncalibRecHitCollection = "ecalGlobalUncalibRecHit:EcalUncalibRecHitsEB"

process.ecalPNDiodeMonitorTask.verbosity = 0
process.ecalPNDiodeMonitorTask.commonParameters.onlineMode = True

process.dqmEnv.subSystemFolder = 'EcalCalibration'
process.dqmSaver.tag = 'EcalCalibration'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'EcalCalibration'
process.dqmSaverPB.runNumber = options.runNumber

### Sequences ###

process.ecalRecoSequence = cms.Sequence((process.ecalGlobalUncalibRecHit+process.ecalDetIdToBeRecovered+process.ecalRecHit))
process.ecalPreRecoSequence = cms.Sequence(process.ecalDigis)

### Paths ###

process.ecalLaserLedPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalLaserLedFilter+process.ecalRecoSequence+process.ecalLaserLedUncalibRecHit+process.ecalLaserLedMonitorTask+process.ecalPNDiodeMonitorTask)
process.ecalTestPulsePath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalTestPulseFilter+process.ecalRecoSequence+process.ecalTestPulseUncalibRecHit+process.ecalTestPulseMonitorTask+process.ecalPNDiodeMonitorTask)
process.ecalPedestalPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalPedestalFilter+process.ecalRecoSequence+process.ecalPedestalMonitorTask+process.ecalPNDiodeMonitorTask)
process.ecalClientPath = cms.Path(process.ecalCalibMonitorClient)

process.dqmEndPath = cms.EndPath(process.dqmEnv)
process.dqmOutputPath = cms.EndPath(process.dqmSaver + process.dqmSaverPB)

### Schedule ###

process.schedule = cms.Schedule(process.ecalLaserLedPath,process.ecalTestPulsePath,process.ecalPedestalPath,process.ecalClientPath,process.dqmEndPath,process.dqmOutputPath)

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
