import FWCore.ParameterSet.Config as cms

# from DQM.EcalBarrelMonitorTasks.EcalMonitorTask_cfi import *

# # placeholder until update propagates to other packages
# from DQM.EcalBarrelMonitorModule.EcalBarrelMonitorModule_cfi import *
# from DQM.EcalEndcapMonitorModule.EcalEndcapMonitorModule_cfi import *

# from DQM.EcalBarrelMonitorTasks.EcalBarrelMonitorTasks_cfi import *
# from DQM.EcalBarrelMonitorTasks.EBHltTask_cfi import *
# from DQM.EcalEndcapMonitorTasks.EcalEndcapMonitorTasks_cfi import *
# from DQM.EcalEndcapMonitorTasks.EEHltTask_cfi import *

# from DQMOffline.Ecal.EBClusterTaskExtras_cfi import *
# from DQMOffline.Ecal.EEClusterTaskExtras_cfi import *

# from DQM.EcalCommon.EcalDQMBinningService_cfi import *

# ecalDQMCollectionTags.EcalRawData = 'ecalDigis:'
# ecalDQMCollectionTags.GainErrors = 'ecalDigis:EcalIntegrityGainErrors'
# ecalDQMCollectionTags.ChIdErrors = 'ecalDigis:EcalIntegrityChIdErrors'
# ecalDQMCollectionTags.GainSwitchErrors = 'ecalDigis:EcalIntegrityGainSwitchErrors'
# ecalDQMCollectionTags.TowerIdErrors = 'ecalDigis:EcalIntegrityTTIdErrors'
# ecalDQMCollectionTags.BlockSizeErrors = 'ecalDigis:EcalIntegrityBlockSizeErrors'
# ecalDQMCollectionTags.MEMTowerIdErrors = 'ecalDigis:EcalIntegrityMemTtIdErrors'
# ecalDQMCollectionTags.MEMBlockSizeErrors = 'ecalDigis:EcalIntegrityMemBlockSizeErrors'
# ecalDQMCollectionTags.MEMChIdErrors = 'ecalDigis:EcalIntegrityMemChIdErrors'
# ecalDQMCollectionTags.MEMGainErrors = 'ecalDigis:EcalIntegrityMemGainErrors'
# ecalDQMCollectionTags.EBSrFlag = 'ecalDigis:'
# ecalDQMCollectionTags.EESrFlag = 'ecalDigis:'
# ecalDQMCollectionTags.EBDigi = 'ecalDigis:ebDigis'
# ecalDQMCollectionTags.EEDigi = 'ecalDigis:eeDigis'
# ecalDQMCollectionTags.PnDiodeDigi = 'ecalDigis:'
# ecalDQMCollectionTags.TrigPrimDigi = 'ecalDigis:EcalTriggerPrimitives'
# ecalDQMCollectionTags.TrigPrimEmulDigi = 'valEcalTriggerPrimitiveDigis'
# ecalDQMCollectionTags.EBUncalibRecHit = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'
# ecalDQMCollectionTags.EEUncalibRecHit = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEE'
# ecalDQMCollectionTags.EBBasicCluster = 'cosmicBasicClusters:CosmicBarrelBasicClusters'
# ecalDQMCollectionTags.EEBasicCluster = 'cosmicBasicClusters:CosmicEndcapBasicClusters'
# ecalDQMCollectionTags.EBSuperCluster = 'cosmicSuperClusters:CosmicBarrelSuperClusters'
# ecalDQMCollectionTags.EESuperCluster = 'cosmicSuperClusters:CosmicEndcapSuperClusters'

# dqmInfoEcal = cms.EDAnalyzer("DQMEventInfo",
#     subSystemFolder = cms.untracked.string('Ecal')
# )

# ecalMonitorTask.tasks = cms.untracked.vstring(
#     "OccupancyTask",
#     "IntegrityTask",
#     "RawDataTask",
#     "PresampleTask",
#     "TrigPrimTask",
#     "ClusterTask",
#     "EnergyTask",
#     "TimingTask"
# )

# ecalMonitorTask.taskParameters.Common.hltTaskMode = 2
# ecalMonitorTask.taskParameters.TrigPrimTask.runOnEmul = False
# ecalMonitorTask.allowMissingCollections = True

# ## standard
# ecal_dqm_source_offline = cms.Sequence(
#     dqmInfoEcal *
#     ecalMonitorTask *
#     ecalBarrelHltTask *
#     ecalBarrelClusterTaskExtras *
#     ecalEndcapClusterTaskExtras
# )

# ## standard with Selective Readout Task and Raw Data Task
# # ecalMonitorTaskSR = ecalMonitorTask.clone()
# # ecalMonitorTaskSR.tasks = cms.untracked.vstring(
# #     "OccupancyTask",
# #     "IntegrityTask",
# #     "RawDataTask",
# #     "PresampleTask",
# #     "TrigPrimTask",
# #     "ClusterTask",
# #     "EnergyTask",
# #     "TimingTask",
# #     "SelectiveReadoutTask"
# # )

# # ecal_dqm_source_offline1 = cms.Sequence(
# #     dqmInfoEcal *
# #     ecalMonitorTaskSR *
# #     ecalBarrelClusterTaskExtras *
# #     ecalEndcapClusterTaskExtras
# # )

# ecalBarrelClusterTaskExtras.BasicClusterCollection = 'cosmicBasicClusters:CosmicBarrelBasicClusters'
# ecalBarrelClusterTaskExtras.SuperClusterCollection = 'cosmicSuperClusters:CosmicBarrelSuperClusters'

# ecalEndcapClusterTaskExtras.BasicClusterCollection = 'cosmicBasicClusters:CosmicEndcapBasicClusters'
# ecalEndcapClusterTaskExtras.SuperClusterCollection = 'cosmicSuperClusters:CosmicEndcapSuperClusters'

# ecalBarrelHltTask.EBDetIdCollection0 = 'ecalDigis:EcalIntegrityDCCSizeErrors'
# ecalBarrelHltTask.EBDetIdCollection1 = 'ecalDigis:EcalIntegrityGainErrors'
# ecalBarrelHltTask.EBDetIdCollection2 = 'ecalDigis:EcalIntegrityChIdErrors'
# ecalBarrelHltTask.EBDetIdCollection3 = 'ecalDigis:EcalIntegrityGainSwitchErrors'
# ecalBarrelHltTask.EcalElectronicsIdCollection1 = 'ecalDigis:EcalIntegrityTTIdErrors'
# ecalBarrelHltTask.EcalElectronicsIdCollection2 = 'ecalDigis:EcalIntegrityBlockSizeErrors'
# ecalBarrelHltTask.EcalElectronicsIdCollection3 = 'ecalDigis:EcalIntegrityMemTtIdErrors'
# ecalBarrelHltTask.EcalElectronicsIdCollection4 = 'ecalDigis:EcalIntegrityMemBlockSizeErrors'
# ecalBarrelHltTask.EcalElectronicsIdCollection5 = 'ecalDigis:EcalIntegrityMemChIdErrors'
# ecalBarrelHltTask.EcalElectronicsIdCollection6 = 'ecalDigis:EcalIntegrityMemGainErrors'


from DQM.EcalBarrelMonitorModule.EcalBarrelMonitorModule_cfi import *
from DQM.EcalEndcapMonitorModule.EcalEndcapMonitorModule_cfi import *

from DQM.EcalBarrelMonitorTasks.EcalBarrelMonitorTasks_cfi import *
from DQM.EcalBarrelMonitorTasks.EBHltTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EcalEndcapMonitorTasks_cfi import *
from DQM.EcalEndcapMonitorTasks.EEHltTask_cfi import *

from DQMOffline.Ecal.EBClusterTaskExtras_cfi import *
from DQMOffline.Ecal.EEClusterTaskExtras_cfi import *

dqmInfoEcal = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Ecal')
)

## standard
eb_dqm_source_offline = cms.Sequence(
    ecalBarrelMonitorModule *
    dqmInfoEcal *
    ecalBarrelOccupancyTask *
    ecalBarrelIntegrityTask *
    ecalBarrelStatusFlagsTask *
    ecalBarrelRawDataTask *
    ecalBarrelPedestalOnlineTask *
    ecalBarrelCosmicTask *
    ecalBarrelTriggerTowerTask *
    ecalBarrelClusterTask *
    ecalBarrelHltTask *
    ecalBarrelClusterTaskExtras
    )

## standard with Selective Readout Task and Raw Data Task
eb_dqm_source_offline1 = cms.Sequence(
    ecalBarrelMonitorModule *
    dqmInfoEcal *
    ecalBarrelOccupancyTask *
    ecalBarrelIntegrityTask *
    ecalBarrelStatusFlagsTask *
    ecalBarrelSelectiveReadoutTask *
    ecalBarrelRawDataTask *
    ecalBarrelPedestalOnlineTask *
    ecalBarrelCosmicTask *
    ecalBarrelTriggerTowerTask *
    ecalBarrelClusterTask *
    ecalBarrelHltTask *    
    ecalBarrelClusterTaskExtras
    )

## standard
ee_dqm_source_offline = cms.Sequence(
    ecalEndcapMonitorModule *
    ecalEndcapOccupancyTask *
    ecalEndcapIntegrityTask *
    ecalEndcapStatusFlagsTask *
    ecalEndcapRawDataTask *
    ecalEndcapPedestalOnlineTask *
    ecalEndcapTriggerTowerTask *
    ecalEndcapCosmicTask *
    ecalEndcapClusterTask *
    ecalEndcapClusterTaskExtras
    )

## standard with Selective Readout Task
ee_dqm_source_offline1 = cms.Sequence(
    ecalEndcapMonitorModule *
    ecalEndcapOccupancyTask *
    ecalEndcapIntegrityTask *
    ecalEndcapStatusFlagsTask *
    ecalEndcapSelectiveReadoutTask *
    ecalEndcapRawDataTask *
    ecalEndcapPedestalOnlineTask *
    ecalEndcapTriggerTowerTask *
    ecalEndcapCosmicTask *
    ecalEndcapClusterTask *
    ecalEndcapClusterTaskExtras
    )

ecal_dqm_source_offline = cms.Sequence(
    eb_dqm_source_offline *
    ee_dqm_source_offline
    )

ecalBarrelMonitorModule.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelMonitorModule.EBDigiCollection = 'ecalDigis:ebDigis'
ecalBarrelMonitorModule.EcalTrigPrimDigiCollection = 'ecalDigis:EcalTriggerPrimitives'
ecalBarrelMonitorModule.verbose = False

ecalEndcapMonitorModule.EcalRawDataCollection = 'ecalDigis:'
ecalEndcapMonitorModule.EEDigiCollection = 'ecalDigis:eeDigis'
ecalEndcapMonitorModule.EcalTrigPrimDigiCollection = 'ecalDigis:EcalTriggerPrimitives'
ecalEndcapMonitorModule.verbose = False

ecalBarrelCosmicTask.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelCosmicTask.EcalUncalibratedRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'

ecalEndcapCosmicTask.EcalRawDataCollection = 'ecalDigis:'
ecalEndcapCosmicTask.EcalUncalibratedRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEE'

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

ecalEndcapIntegrityTask.EEDetIdCollection0 = 'ecalDigis:EcalIntegrityDCCSizeErrors'
ecalEndcapIntegrityTask.EEDetIdCollection1 = 'ecalDigis:EcalIntegrityGainErrors'
ecalEndcapIntegrityTask.EEDetIdCollection2 = 'ecalDigis:EcalIntegrityChIdErrors'
ecalEndcapIntegrityTask.EEDetIdCollection3 = 'ecalDigis:EcalIntegrityGainSwitchErrors'
ecalEndcapIntegrityTask.EcalElectronicsIdCollection1 = 'ecalDigis:EcalIntegrityTTIdErrors'
ecalEndcapIntegrityTask.EcalElectronicsIdCollection2 = 'ecalDigis:EcalIntegrityBlockSizeErrors'
ecalEndcapIntegrityTask.EcalElectronicsIdCollection3 = 'ecalDigis:EcalIntegrityMemTtIdErrors'
ecalEndcapIntegrityTask.EcalElectronicsIdCollection4 = 'ecalDigis:EcalIntegrityMemBlockSizeErrors'
ecalEndcapIntegrityTask.EcalElectronicsIdCollection5 = 'ecalDigis:EcalIntegrityMemChIdErrors'
ecalEndcapIntegrityTask.EcalElectronicsIdCollection6 = 'ecalDigis:EcalIntegrityMemGainErrors'

ecalBarrelOccupancyTask.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelOccupancyTask.EBDigiCollection = 'ecalDigis:ebDigis'
ecalBarrelOccupancyTask.EcalPnDiodeDigiCollection = 'ecalDigis:'
ecalBarrelOccupancyTask.EcalTrigPrimDigiCollection = 'ecalDigis:EcalTriggerPrimitives'

ecalEndcapOccupancyTask.EcalRawDataCollection = 'ecalDigis:'
ecalEndcapOccupancyTask.EEDigiCollection = 'ecalDigis:eeDigis'
ecalEndcapOccupancyTask.EcalPnDiodeDigiCollection = 'ecalDigis:'
ecalEndcapOccupancyTask.EcalTrigPrimDigiCollection = 'ecalDigis:EcalTriggerPrimitives'

ecalBarrelPedestalOnlineTask.EBDigiCollection = 'ecalDigis:ebDigis'

ecalEndcapPedestalOnlineTask.EEDigiCollection = 'ecalDigis:eeDigis'

ecalBarrelStatusFlagsTask.EcalRawDataCollection = 'ecalDigis:'

ecalEndcapStatusFlagsTask.EcalRawDataCollection = 'ecalDigis:'

ecalBarrelRawDataTask.EcalRawDataCollection = 'ecalDigis:'

ecalEndcapRawDataTask.EcalRawDataCollection = 'ecalDigis:'

ecalBarrelSelectiveReadoutTask.EBDigiCollection = 'ecalDigis:ebDigis'
ecalBarrelSelectiveReadoutTask.EBSRFlagCollection = 'ecalDigis:'
ecalBarrelSelectiveReadoutTask.EcalTrigPrimDigiCollection = 'ecalDigis:EcalTriggerPrimitives'

ecalEndcapSelectiveReadoutTask.EEDigiCollection = 'ecalDigis:eeDigis'
ecalEndcapSelectiveReadoutTask.EESRFlagCollection = 'ecalDigis:'
ecalEndcapSelectiveReadoutTask.EcalTrigPrimDigiCollection = 'ecalDigis:EcalTriggerPrimitives'

ecalBarrelTimingTask.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelTimingTask.EcalRecHitCollection = 'ecalRecHit:EcalRecHitsEB'

ecalEndcapTimingTask.EcalRawDataCollection = 'ecalDigis:'
ecalEndcapTimingTask.EcalRecHitCollection = 'ecalRecHit:EcalRecHitsEE'

ecalBarrelTriggerTowerTask.EcalTrigPrimDigiCollectionReal = 'ecalDigis:EcalTriggerPrimitives'
ecalBarrelTriggerTowerTask.EBDigiCollection = 'ecalDigis:ebDigis'

ecalEndcapTriggerTowerTask.EcalTrigPrimDigiCollectionReal = 'ecalDigis:EcalTriggerPrimitives'
ecalEndcapTriggerTowerTask.EEDigiCollection = 'ecalDigis:eeDigis'

# to be used if the TP emulator _is_not_ in the path
#ecalBarrelTriggerTowerTask.EcalTrigPrimDigiCollectionEmul = 'ecalDigis:EcalTriggerPrimitives'
#ecalEndcapTriggerTowerTask.EcalTrigPrimDigiCollectionEmul = 'ecalDigis:EcalTriggerPrimitives'

# to be used if the TP emulator _is_ in the path
ecalBarrelTriggerTowerTask.EcalTrigPrimDigiCollectionEmul = 'valEcalTriggerPrimitiveDigis'
ecalEndcapTriggerTowerTask.EcalTrigPrimDigiCollectionEmul = 'valEcalTriggerPrimitiveDigis'

ecalBarrelClusterTask.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelClusterTask.BasicClusterCollection = 'cosmicBasicClusters:CosmicBarrelBasicClusters'
ecalBarrelClusterTask.SuperClusterCollection = 'cosmicSuperClusters:CosmicBarrelSuperClusters'

ecalEndcapClusterTask.EcalRawDataCollection = 'ecalDigis:'
ecalEndcapClusterTask.BasicClusterCollection = 'cosmicBasicClusters:CosmicEndcapBasicClusters'
ecalEndcapClusterTask.SuperClusterCollection = 'cosmicSuperClusters:CosmicEndcapSuperClusters'

ecalBarrelClusterTaskExtras.BasicClusterCollection = 'cosmicBasicClusters:CosmicBarrelBasicClusters'
ecalBarrelClusterTaskExtras.SuperClusterCollection = 'cosmicSuperClusters:CosmicBarrelSuperClusters'

ecalEndcapClusterTaskExtras.BasicClusterCollection = 'cosmicBasicClusters:CosmicEndcapBasicClusters'
ecalEndcapClusterTaskExtras.SuperClusterCollection = 'cosmicSuperClusters:CosmicEndcapSuperClusters'

ecalBarrelHltTask.EBDetIdCollection0 = 'ecalDigis:EcalIntegrityDCCSizeErrors'
ecalBarrelHltTask.EBDetIdCollection1 = 'ecalDigis:EcalIntegrityGainErrors'
ecalBarrelHltTask.EBDetIdCollection2 = 'ecalDigis:EcalIntegrityChIdErrors'
ecalBarrelHltTask.EBDetIdCollection3 = 'ecalDigis:EcalIntegrityGainSwitchErrors'
ecalBarrelHltTask.EcalElectronicsIdCollection1 = 'ecalDigis:EcalIntegrityTTIdErrors'
ecalBarrelHltTask.EcalElectronicsIdCollection2 = 'ecalDigis:EcalIntegrityBlockSizeErrors'
ecalBarrelHltTask.EcalElectronicsIdCollection3 = 'ecalDigis:EcalIntegrityMemTtIdErrors'
ecalBarrelHltTask.EcalElectronicsIdCollection4 = 'ecalDigis:EcalIntegrityMemBlockSizeErrors'
ecalBarrelHltTask.EcalElectronicsIdCollection5 = 'ecalDigis:EcalIntegrityMemChIdErrors'
ecalBarrelHltTask.EcalElectronicsIdCollection6 = 'ecalDigis:EcalIntegrityMemGainErrors'

