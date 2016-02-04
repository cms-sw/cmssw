import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorModule.EcalBarrelMonitorModule_cfi import *
from DQM.EcalEndcapMonitorModule.EcalEndcapMonitorModule_cfi import *

from DQM.EcalBarrelMonitorTasks.EcalBarrelMonitorTasks_cfi import *
from DQM.EcalBarrelMonitorTasks.EBHltTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EcalEndcapMonitorTasks_cfi import *
from DQM.EcalEndcapMonitorTasks.EEHltTask_cfi import *

from DQMOffline.Ecal.EBClusterTaskExtras_cfi import *
from DQMOffline.Ecal.EEClusterTaskExtras_cfi import *

dqmInfoEB = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('EcalBarrel')
)

dqmInfoEE = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('EcalEndcap')
)

## standard
eb_dqm_source_offline = cms.Sequence(ecalBarrelMonitorModule*dqmInfoEB*ecalBarrelOccupancyTask*ecalBarrelIntegrityTask*ecalBarrelStatusFlagsTask*ecalBarrelRawDataTask*ecalBarrelPedestalOnlineTask*ecalBarrelTriggerTowerTask*ecalBarrelClusterTask*ecalBarrelHltTask*ecalBarrelClusterTaskExtras)

## standard with Selective Readout Task
eb_dqm_source_offline1 = cms.Sequence(ecalBarrelMonitorModule*dqmInfoEB*ecalBarrelOccupancyTask*ecalBarrelIntegrityTask*ecalBarrelStatusFlagsTask*ecalBarrelSelectiveReadoutTask*ecalBarrelRawDataTask*ecalBarrelPedestalOnlineTask*ecalBarrelTriggerTowerTask*ecalBarrelClusterTask*ecalBarrelHltTask*ecalBarrelClusterTaskExtras)

## standard
ee_dqm_source_offline = cms.Sequence(ecalEndcapMonitorModule*dqmInfoEE*ecalEndcapOccupancyTask*ecalEndcapIntegrityTask*ecalEndcapStatusFlagsTask*ecalEndcapRawDataTask*ecalEndcapPedestalOnlineTask*ecalEndcapTriggerTowerTask*ecalEndcapClusterTask*ecalEndcapHltTask*ecalEndcapClusterTaskExtras)

## standard with Selective Readout Task
ee_dqm_source_offline1 = cms.Sequence(ecalEndcapMonitorModule*dqmInfoEE*ecalEndcapOccupancyTask*ecalEndcapIntegrityTask*ecalEndcapStatusFlagsTask*ecalEndcapSelectiveReadoutTask*ecalEndcapRawDataTask*ecalEndcapPedestalOnlineTask*ecalEndcapTriggerTowerTask*ecalEndcapClusterTask*ecalEndcapHltTask*ecalEndcapClusterTaskExtras)

ecal_dqm_source_offline = cms.Sequence(eb_dqm_source_offline*ee_dqm_source_offline)

ecalBarrelMonitorModule.EcalRawDataCollection = 'ecalDigis:'
ecalBarrelMonitorModule.EBDigiCollection = 'ecalDigis:ebDigis'
ecalBarrelMonitorModule.EcalTrigPrimDigiCollection = 'ecalDigis:EcalTriggerPrimitives'
ecalBarrelMonitorModule.verbose = False

ecalEndcapMonitorModule.EcalRawDataCollection = 'ecalDigis:'
ecalEndcapMonitorModule.EEDigiCollection = 'ecalDigis:eeDigis'
ecalEndcapMonitorModule.EcalTrigPrimDigiCollection = 'ecalDigis:EcalTriggerPrimitives'
ecalEndcapMonitorModule.verbose = False

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
ecalBarrelClusterTask.BasicClusterCollection = 'hybridSuperClusters:hybridBarrelBasicClusters'
ecalBarrelClusterTask.SuperClusterCollection = 'correctedHybridSuperClusters:'

ecalEndcapClusterTask.EcalRawDataCollection = 'ecalDigis:'
ecalEndcapClusterTask.BasicClusterCollection = 'multi5x5SuperClusters:multi5x5EndcapBasicClusters'
ecalEndcapClusterTask.SuperClusterCollection = 'multi5x5SuperClusters:multi5x5EndcapSuperClusters'

ecalBarrelClusterTaskExtras.BasicClusterCollection = 'hybridSuperClusters:hybridBarrelBasicClusters'
ecalBarrelClusterTaskExtras.SuperClusterCollection = 'correctedHybridSuperClusters:'

ecalEndcapClusterTaskExtras.BasicClusterCollection = 'multi5x5SuperClusters:multi5x5EndcapBasicClusters'
ecalEndcapClusterTaskExtras.SuperClusterCollection = 'multi5x5SuperClusters:multi5x5EndcapSuperClusters'

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

ecalEndcapHltTask.EEDetIdCollection0 = 'ecalDigis:EcalIntegrityDCCSizeErrors'
ecalEndcapHltTask.EEDetIdCollection1 = 'ecalDigis:EcalIntegrityGainErrors'
ecalEndcapHltTask.EEDetIdCollection2 = 'ecalDigis:EcalIntegrityChIdErrors'
ecalEndcapHltTask.EEDetIdCollection3 = 'ecalDigis:EcalIntegrityGainSwitchErrors'
ecalEndcapHltTask.EcalElectronicsIdCollection1 = 'ecalDigis:EcalIntegrityTTIdErrors'
ecalEndcapHltTask.EcalElectronicsIdCollection2 = 'ecalDigis:EcalIntegrityBlockSizeErrors'
ecalEndcapHltTask.EcalElectronicsIdCollection3 = 'ecalDigis:EcalIntegrityMemTtIdErrors'
ecalEndcapHltTask.EcalElectronicsIdCollection4 = 'ecalDigis:EcalIntegrityMemBlockSizeErrors'
ecalEndcapHltTask.EcalElectronicsIdCollection5 = 'ecalDigis:EcalIntegrityMemChIdErrors'
ecalEndcapHltTask.EcalElectronicsIdCollection6 = 'ecalDigis:EcalIntegrityMemGainErrors'
