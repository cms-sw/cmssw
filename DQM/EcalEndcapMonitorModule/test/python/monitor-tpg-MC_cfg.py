import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALDQM")

import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()

#import RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi
#process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi.ecalGlobalUncalibRecHit.clone()

process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_cff")
process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi")

import SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi
process.simEcalTriggerPrimitiveDigis2 = SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi.simEcalTriggerPrimitiveDigis.clone()

process.load("DQM.EcalEndcapMonitorModule.EcalEndcapMonitorModule_cfi")

process.load("DQM.EcalEndcapMonitorTasks.EcalEndcapMonitorTasks_cfi")

process.load("DQM.EcalEndcapMonitorTasks.mergeRuns_cff")

process.load("DQM.EcalEndcapMonitorClient.EcalEndcapMonitorClient_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("FWCore.Modules.preScaler_cfi")

process.dqmInfoEE = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('EcalEndcap')
)

process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    dirName = cms.untracked.string('.'),
    convention = cms.untracked.string('Online')
)

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(150)
    input = cms.untracked.int32(300)
)
process.source = cms.Source("PoolSource",
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
#---
    setRunNumber = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('/store/user/dellaric/data/relval_zee_310.root')
#---
)

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.EcalTrivialConditionRetriever.adcToGeVEBConstant = cms.untracked.double(0.035)
process.EcalTrivialConditionRetriever.adcToGeVEEConstant = cms.untracked.double(0.060)
process.EcalTrivialConditionRetriever.getWeightsFromFile = False
process.EcalTrivialConditionRetriever.pedWeights = cms.untracked.vdouble(0.333, 0.333, 0.333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
process.EcalTrivialConditionRetriever.pedWeightsAft = cms.untracked.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
process.EcalTrivialConditionRetriever.amplWeights = cms.untracked.vdouble(-0.333, -0.333, -0.333, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
process.EcalTrivialConditionRetriever.amplWeightsAftGain = cms.untracked.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
process.EcalTrivialConditionRetriever.jittWeights = cms.untracked.vdouble(0.041, 0.041, 0.041, 0.0, 1.325, -0.05, -0.504, -0.502, -0.390, 0.0)
process.EcalTrivialConditionRetriever.jittWeightsAft = cms.untracked.vdouble(0.0, 0.0, 0.0, 0.0, 1.098, -0.046, -0.416, -0.419, -0.337, 0.0)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True),
        noTimeStamps = cms.untracked.bool(True),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalEndcapMonitorModule = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('EcalEndcapMonitorModule'),
    destinations = cms.untracked.vstring('cout')
)

process.preScaler.prescaleFactor = 1

process.ecalDataSequence = cms.Sequence(process.preScaler*process.ecalUncalibHit*process.ecalRecHit*process.simEcalTriggerPrimitiveDigis*process.simEcalTriggerPrimitiveDigis2*process.hybridClusteringSequence*process.multi5x5BasicClusters*process.multi5x5SuperClusters)
process.ecalEndcapMonitorSequence = cms.Sequence(process.ecalEndcapMonitorModule*process.dqmInfoEE*process.ecalEndcapMonitorClient)

process.p = cms.Path(process.ecalDataSequence*process.ecalEndcapMonitorSequence*process.dqmSaver)
process.q = cms.EndPath(process.ecalEndcapDefaultTasksSequence*process.ecalEndcapCosmicTask*process.ecalEndcapClusterTask)

process.ecalUncalibHit.MinAmplBarrel = 12.
process.ecalUncalibHit.MinAmplEndcap = 16.
process.ecalUncalibHit.EBdigiCollection = 'simEcalDigis:ebDigis'
process.ecalUncalibHit.EEdigiCollection = 'simEcalDigis:eeDigis'

process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'

process.ecalEndcapMonitorModule.mergeRuns = True
process.ecalEndcapMonitorModule.EEDigiCollection = 'simEcalDigis:eeDigis'
process.ecalEndcapMonitorModule.runType = 13 # PHYSICS_GLOBAL

process.simEcalTriggerPrimitiveDigis.Label = 'simEcalDigis'
process.simEcalTriggerPrimitiveDigis.InstanceEB = 'ebDigis'
process.simEcalTriggerPrimitiveDigis.InstanceEE = 'eeDigis'

process.simEcalTriggerPrimitiveDigis2.Label = 'simEcalDigis'
process.simEcalTriggerPrimitiveDigis2.InstanceEB = 'ebDigis'
process.simEcalTriggerPrimitiveDigis2.InstanceEE = 'eeDigis'

process.ecalEndcapTriggerTowerTask.EcalTrigPrimDigiCollectionReal = 'simEcalTriggerPrimitiveDigis2'

process.ecalEndcapMonitorModule.EcalTrigPrimDigiCollection = 'simEcalTriggerPrimitiveDigis2'

process.ecalEndcapOccupancyTask.EEDigiCollection = 'simEcalDigis:eeDigis'
process.ecalEndcapOccupancyTask.EcalTrigPrimDigiCollection = 'simEcalTriggerPrimitiveDigis'

process.ecalEndcapPedestalOnlineTask.EEDigiCollection = 'simEcalDigis:eeDigis'

process.ecalEndcapTriggerTowerTask.EEDigiCollection = 'simEcalDigis:eeDigis'

process.ecalEndcapMonitorClient.mergeRuns = True
process.ecalEndcapMonitorClient.location = 'H4'
process.ecalEndcapMonitorClient.enabledClients = ['Integrity', 'Occupancy', 'PedestalOnline', 'Cosmic', 'Timing', 'TriggerTower', 'Cluster', 'Summary']

process.DQM.collectorHost = ''

