import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALDQM")

import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()

#import RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi
#process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi.ecalWeightUncalibRecHit.clone()

process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("DQM.EcalBarrelMonitorModule.EcalBarrelMonitorModule_cfi")

process.load("DQM.EcalBarrelMonitorTasks.EcalBarrelMonitorTasks_cfi")

process.load("DQM.EcalBarrelMonitorTasks.mergeRuns_cff")

process.load("DQM.EcalBarrelMonitorClient.EcalBarrelMonitorClient_cfi")

process.load("RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.preScaler = cms.EDFilter("Prescaler",
    prescaleFactor = cms.int32(1)
)

process.dqmInfoEB = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('EcalBarrel')
)

process.dqmQTestEB = cms.EDAnalyzer("QualityTester",
    reportThreshold = cms.untracked.string('red'),
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/EcalBarrelMonitorModule/test/data/EcalBarrelQualityTests.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    dirName = cms.untracked.string('.'),
    convention = cms.untracked.string('Online')
)

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(150)
#    input = cms.untracked.int32(300)
)
process.source = cms.Source("PoolSource",
#---
#    fileNames = cms.untracked.vstring('/store/users/dellaric/data/relval_zee.root')
#---
    fileNames = cms.untracked.vstring('/store/users/dellaric/data/relval_zee.root', 
        '/store/users/dellaric/data/relval_zee.root', 
        '/store/users/dellaric/data/relval_zee.root', 
        '/store/users/dellaric/data/relval_zee.root', 
        '/store/users/dellaric/data/relval_zee.root', 
        '/store/users/dellaric/data/relval_zee.root', 
        '/store/users/dellaric/data/relval_zee.root', 
        '/store/users/dellaric/data/relval_zee.root', 
        '/store/users/dellaric/data/relval_zee.root', 
        '/store/users/dellaric/data/relval_zee.root', 
        '/store/users/dellaric/data/relval_zee.root', 
        '/store/users/dellaric/data/relval_zee.root')
#---
)

process.EcalTrivialConditionRetriever = cms.ESSource("EcalTrivialConditionRetriever",
    adcToGeVEBConstant = cms.untracked.double(0.035),
    adcToGeVEEConstant = cms.untracked.double(0.06),
    pedWeights = cms.untracked.vdouble(0.333, 0.333, 0.333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    amplWeights = cms.untracked.vdouble(-0.333, -0.333, -0.333, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    jittWeights = cms.untracked.vdouble(0.041, 0.041, 0.041, 0.0, 1.325, -0.05, -0.504, -0.502, -0.390, 0.0)
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True),
        noTimeStamps = cms.untracked.bool(True),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalBarrelMonitorModule = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('EcalBarrelMonitorModule'),
    destinations = cms.untracked.vstring('cout')
)

process.ecalDataSequence = cms.Sequence(process.preScaler*process.ecalUncalibHit*process.ecalRecHit*process.hybridSuperClusters*process.correctedHybridSuperClusters*process.multi5x5BasicClusters*process.multi5x5SuperClusters)
process.ecalBarrelMonitorSequence = cms.Sequence(process.ecalBarrelMonitorModule*process.dqmInfoEB*process.ecalBarrelMonitorClient*process.dqmQTestEB)

process.p = cms.Path(process.ecalDataSequence*process.ecalBarrelMonitorSequence*process.dqmSaver)
process.q = cms.EndPath(process.ecalBarrelDefaultTasksSequence*process.ecalBarrelClusterTask)

process.ecalUncalibHit.MinAmplBarrel = 12.
process.ecalUncalibHit.MinAmplEndcap = 16.
process.ecalUncalibHit.EBdigiCollection = 'simEcalDigis:ebDigis'
process.ecalUncalibHit.EEdigiCollection = 'simEcalDigis:eeDigis'

process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'

process.ecalBarrelMonitorModule.mergeRuns = True
process.ecalBarrelMonitorModule.EBDigiCollection = 'simEcalDigis:ebDigis'
process.ecalBarrelMonitorModule.runType = 3 # MTCC/PHYSICS
process.ecalBarrelMonitorModule.EcalTrigPrimDigiCollection = 'ecalTriggerPrimitiveDigis'

process.ecalBarrelOccupancyTask.EBDigiCollection = 'simEcalDigis:ebDigis'
process.ecalBarrelOccupancyTask.EcalTrigPrimDigiCollection = 'ecalTriggerPrimitiveDigis'

process.ecalBarrelPedestalOnlineTask.EBDigiCollection = 'simEcalDigis:ebDigis'

process.ecalBarrelTriggerTowerTask.EBDigiCollection = 'simEcalDigis:ebDigis'

process.ecalBarrelMonitorClient.maskFile = '../data/maskfile-EB.dat'
process.ecalBarrelMonitorClient.mergeRuns = True
process.ecalBarrelMonitorClient.location = 'H4'
process.ecalBarrelMonitorClient.enabledClients = ['Integrity', 'Occupancy', 'PedestalOnline', 'Timing', 'Cluster', 'Summary']

