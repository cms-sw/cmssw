import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorModule.EcalBarrelMonitorModule_cfi import *
from DQM.EcalBarrelMonitorTasks.EcalBarrelMonitorTasks_cfi import *

dqmInfoEB = cms.EDFilter("DQMEventInfo",
    subSystemFolder = cms.untracked.string('EcalBarrel')
)

ecal_dqm_source_offline1 = cms.Sequence(ecalBarrelMonitorModule*dqmInfoEB*ecalBarrelIntegrityTask)
ecal_dqm_source_offline2 = cms.Sequence(ecalBarrelMonitorModule*dqmInfoEB*ecalBarrelIntegrityTask*ecalBarrelStatusFlagsTask)
ecal_dqm_source_offline3 = cms.Sequence(ecalBarrelMonitorModule*dqmInfoEB*ecalBarrelIntegrityTask*ecalBarrelStatusFlagsTask*ecalBarrelCosmicTask)
ecal_dqm_source_offline4 = cms.Sequence(ecalBarrelMonitorModule*dqmInfoEB*ecalBarrelOccupancyTask*ecalBarrelIntegrityTask*ecalBarrelStatusFlagsTask*ecalBarrelCosmicTask)
ecal_dqm_source_offline5 = cms.Sequence(ecalBarrelMonitorModule*dqmInfoEB*ecalBarrelOccupancyTask*ecalBarrelIntegrityTask*ecalBarrelStatusFlagsTask*ecalBarrelPedestalOnlineTask*ecalBarrelCosmicTask)
ecal_dqm_source_offline6 = cms.Sequence(ecalBarrelMonitorModule*dqmInfoEB*ecalBarrelOccupancyTask*ecalBarrelIntegrityTask*ecalBarrelStatusFlagsTask*ecalBarrelPedestalOnlineTask*ecalBarrelTriggerTowerTask*ecalBarrelCosmicTask)
ecal_dqm_source_offline7 = cms.Sequence(ecalBarrelMonitorModule*dqmInfoEB*ecalBarrelOccupancyTask*ecalBarrelIntegrityTask*ecalBarrelStatusFlagsTask*ecalBarrelPedestalOnlineTask*ecalBarrelTriggerTowerTask*ecalBarrelCosmicTask*ecalBarrelClusterTask)
ecal_dqm_source_offline9 = cms.Sequence(ecalBarrelMonitorModule*dqmInfoEB*ecalBarrelOccupancyTask*ecalBarrelIntegrityTask*ecalBarrelStatusFlagsTask*ecalBarrelLaserTask*ecalBarrelPedestalOnlineTask*ecalBarrelPedestalTask*ecalBarrelTestPulseTask*ecalBarrelTriggerTowerTask*ecalBarrelTimingTask*ecalBarrelCosmicTask*ecalBarrelClusterTask)

ecal_dqm_source_offline = cms.Sequence(ecal_dqm_source_offline1)

ecalBarrelMonitorModule.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelMonitorModule.EBDigiCollection = 'ecalDigis:ebDigis'
ecalBarrelMonitorModule.EcalTrigPrimDigiCollection = 'ecalDigis:EcalTriggerPrimitives'
ecalBarrelMonitorModule.verbose = False

ecalBarrelCosmicTask.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelCosmicTask.EcalUncalibratedRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'

ecalBarrelIntegrityTask.EBDetIdCollection0 = 'ecalDigis:EcalIntegrityDCCSizeErrors'
ecalBarrelIntegrityTask.EBDetIdCollection1 = 'ecalDigis:EcalIntegrityGainErrors'
ecalBarrelIntegrityTask.EBDetIdCollection2 = 'ecalDigis:EcalIntegrityChIdErrors'
ecalBarrelIntegrityTask.EBDetIdCollection3 = 'ecalDigis:EcalIntegrityGainSwitchErrors'
ecalBarrelIntegrityTask.EcalElectronicsIdCollection1 = 'ecalDigis:EcalIntegrityTTIdErrors'
ecalBarrelIntegrityTask.EcalElectronicsIdCollection2 = 'ecalDigis:EcalIntegrityBlockSizeErrors'
ecalBarrelIntegrityTask.EcalElectronicsIdCollection3 = 'ecalDigis:EcalIntegrityMemTtIdErrors'
ecalBarrelIntegrityTask.EcalElectronicsIdCollection4 = 'ecalDigis:EcalIntegrityMemBlockSizeErrors'
ecalBarrelIntegrityTask.EcalElectronicsIdCollection5 = 'ecalDigis:EcalIntegrityMemChIdErrors'
ecalBarrelIntegrityTask.EcalElectronicsIdCollection6 = 'ecalDigis:EcalIntegrityMemGainErrors'

ecalBarrelLaserTask.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelLaserTask.EBDigiCollection = 'ecalDigis:ebDigis'
ecalBarrelLaserTask.EcalPnDiodeDigiCollection = 'ecalDigis:'
ecalBarrelLaserTask.EcalUncalibratedRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'

ecalBarrelOccupancyTask.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelOccupancyTask.EBDigiCollection = 'ecalDigis:ebDigis'
ecalBarrelOccupancyTask.EcalPnDiodeDigiCollection = 'ecalDigis:'
ecalBarrelOccupancyTask.EcalTrigPrimDigiCollection = 'ecalDigis:EcalTriggerPrimitives'

ecalBarrelPedestalOnlineTask.EBDigiCollection = 'ecalDigis:ebDigis'

ecalBarrelPedestalTask.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelPedestalTask.EBDigiCollection = 'ecalDigis:ebDigis'
ecalBarrelPedestalTask.EcalPnDiodeDigiCollection = 'ecalDigis:'

ecalBarrelStatusFlagsTask.EcalRawDataCollection = 'ecalDigis:'

ecalBarrelTestPulseTask.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelTestPulseTask.EBDigiCollection = 'ecalDigis:ebDigis'
ecalBarrelTestPulseTask.EcalPnDiodeDigiCollection = 'ecalDigis:'
ecalBarrelTestPulseTask.EcalUncalibratedRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'

ecalBarrelTimingTask.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelTimingTask.EcalUncalibratedRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'

ecalBarrelTriggerTowerTask.EcalTrigPrimDigiCollectionReal = 'ecalDigis:EcalTriggerPrimitives'

# to be used if the TP emulator _is_not_ in the path
ecalBarrelTriggerTowerTask.EcalTrigPrimDigiCollectionEmul = 'ecalDigis:EcalTriggerPrimitives'

# to be used if the TP emulator _is_ in the path
#ecalBarrelTriggerTowerTask.EcalTrigPrimDigiCollectionEmul = 'valEcalTriggerPrimitiveDigis'

ecalBarrelClusterTask.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelClusterTask.BasicClusterCollection = 'cosmicBasicClusters:CosmicBarrelBasicClusters'
ecalBarrelClusterTask.SuperClusterCollection = 'hybridSuperClusters:'

