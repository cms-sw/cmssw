import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALDQM")

process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHit2 = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()

import RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi.ecalGlobalUncalibRecHit.clone()

process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

process.load("DQM.EcalEndcapMonitorModule.EcalEndcapMonitorModule_cfi")

process.load("DQM.EcalEndcapMonitorTasks.EcalEndcapMonitorTasks_cfi")

process.load("DQM.EcalEndcapMonitorClient.EcalEndcapMonitorClient_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("FWCore.Modules.preScaler_cfi")

process.dqmInfoEE = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('EcalEndcap')
)

process.dqmQTestEE = cms.EDAnalyzer("QualityTester",
    reportThreshold = cms.untracked.string('red'),
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/EcalEndcapMonitorModule/test/data/EcalEndcapQualityTests.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    dirName = cms.untracked.string('.'),
    convention = cms.untracked.string('Online')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(150)
)

process.source = cms.Source("PoolSource",
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
#---
    fileNames = cms.untracked.vstring('/store/data/Commissioning08/Calo/RAW/v1/000/069/382/0A023003-3BAB-DD11-B4D0-000423D6B5C4.root')
#---
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR10_P_V12::All"

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True),
        noTimeStamps = cms.untracked.bool(True),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalTBRawToDigi = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiTriggerType = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiTpg = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiNumTowerBlocks = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiTowerId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiTowerSize = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiChId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiGainZero = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiGainSwitch = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalTBRawToDigiDccBlockSize = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigi = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTriggerType = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTpg = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiNumTowerBlocks = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTowerId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTowerSize = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiChId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiGainZero = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiGainSwitch = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiDccBlockSize = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemBlock = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemTowerId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemChId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemGain = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTCC = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiSRP = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalDCCHeaderRuntypeDecoder = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalEndcapMonitorModule = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('EcalTBRawToDigi', 
                                       'EcalTBRawToDigiTriggerType', 
                                       'EcalTBRawToDigiTpg', 
                                       'EcalTBRawToDigiNumTowerBlocks', 
                                       'EcalTBRawToDigiTowerId', 
                                       'EcalTBRawToDigiTowerSize', 
                                       'EcalTBRawToDigiChId', 
                                       'EcalTBRawToDigiGainZero', 
                                       'EcalTBRawToDigiGainSwitch', 
                                       'EcalTBRawToDigiDccBlockSize', 
                                       'EcalRawToDigi', 
                                       'EcalRawToDigiTriggerType', 
                                       'EcalRawToDigiTpg', 
                                       'EcalRawToDigiNumTowerBlocks', 
                                       'EcalRawToDigiTowerId', 
                                       'EcalRawToDigiTowerSize', 
                                       'EcalRawToDigiChId', 
                                       'EcalRawToDigiGainZero', 
                                       'EcalRawToDigiGainSwitch', 
                                       'EcalRawToDigiDccBlockSize', 
                                       'EcalRawToDigiMemBlock', 
                                       'EcalRawToDigiMemTowerId', 
                                       'EcalRawToDigiMemChId', 
                                       'EcalRawToDigiMemGain', 
                                       'EcalRawToDigiTCC', 
                                       'EcalRawToDigiSRP', 
                                       'EcalDCCHeaderRuntypeDecoder', 
                                       'EcalEndcapMonitorModule'),
    destinations = cms.untracked.vstring('cout')
)

process.preScaler.prescaleFactor = 1

process.ecalDataSequence = cms.Sequence(process.preScaler*process.ecalEBunpacker*process.ecalUncalibHit*process.ecalUncalibHit2*process.ecalRecHit)
process.ecalEndcapMonitorSequence = cms.Sequence(process.ecalEndcapMonitorModule*process.dqmInfoEE*process.ecalEndcapMonitorClient*process.dqmQTestEE)

process.p = cms.Path(process.ecalDataSequence*process.ecalEndcapMonitorSequence*process.dqmSaver)
process.q = cms.EndPath(process.ecalEndcapCertificationSequence*process.ecalEndcapCosmicTasksSequence)

process.ecalUncalibHit2.MinAmplBarrel = 12.
process.ecalUncalibHit2.MinAmplEndcap = 16.
process.ecalUncalibHit2.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit2.EEdigiCollection = 'ecalEBunpacker:eeDigis'

process.ecalUncalibHit.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit.EEdigiCollection = 'ecalEBunpacker:eeDigis'

process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit2:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit2:EcalUncalibRecHitsEE'

process.ecalEndcapCosmicTask.EcalUncalibratedRecHitCollection = 'ecalUncalibHit2:EcalUncalibRecHitsEE'

process.ecalEndcapLaserTask.EcalUncalibratedRecHitCollection = 'ecalUncalibHit2:EcalUncalibRecHitsEE'

process.ecalEndcapLedTask.EcalUncalibratedRecHitCollection = 'ecalUncalibHit2:EcalUncalibRecHitsEE'

process.ecalEndcapTimingTask.EcalRecHitCollection = 'ecalRecHit:EcalRecHitsEE'

process.ecalEndcapMonitorClient.location = 'P5_Co'
#process.ecalEndcapMonitorClient.superModules = [34]

process.DQM.collectorHost = ''

