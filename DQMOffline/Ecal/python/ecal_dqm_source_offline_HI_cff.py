import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorTasks.EcalMonitorTask_cfi import *

# placeholder until update propagates to other packages
from DQM.EcalBarrelMonitorModule.EcalBarrelMonitorModule_cfi import *
from DQM.EcalEndcapMonitorModule.EcalEndcapMonitorModule_cfi import *

from DQM.EcalBarrelMonitorTasks.EcalBarrelMonitorTasks_cfi import *
from DQM.EcalBarrelMonitorTasks.EBHltTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EcalEndcapMonitorTasks_cfi import *
from DQM.EcalEndcapMonitorTasks.EEHltTask_cfi import *

from DQMOffline.Ecal.EBClusterTaskExtras_cfi import *
from DQMOffline.Ecal.EEClusterTaskExtras_cfi import *

from DQMOffline.Ecal.EcalZmassTask_cfi import *

from DQM.EcalCommon.EcalDQMBinningService_cfi import *

ecalDQMCollectionTags.EcalRawData = 'ecalDigis:'
ecalDQMCollectionTags.GainErrors = 'ecalDigis:EcalIntegrityGainErrors'
ecalDQMCollectionTags.ChIdErrors = 'ecalDigis:EcalIntegrityChIdErrors'
ecalDQMCollectionTags.GainSwitchErrors = 'ecalDigis:EcalIntegrityGainSwitchErrors'
ecalDQMCollectionTags.TowerIdErrors = 'ecalDigis:EcalIntegrityTTIdErrors'
ecalDQMCollectionTags.BlockSizeErrors = 'ecalDigis:EcalIntegrityBlockSizeErrors'
ecalDQMCollectionTags.MEMTowerIdErrors = 'ecalDigis:EcalIntegrityMemTtIdErrors'
ecalDQMCollectionTags.MEMBlockSizeErrors = 'ecalDigis:EcalIntegrityMemBlockSizeErrors'
ecalDQMCollectionTags.MEMChIdErrors = 'ecalDigis:EcalIntegrityMemChIdErrors'
ecalDQMCollectionTags.MEMGainErrors = 'ecalDigis:EcalIntegrityMemGainErrors'
ecalDQMCollectionTags.EBSrFlag = 'ecalDigis:'
ecalDQMCollectionTags.EESrFlag = 'ecalDigis:'
ecalDQMCollectionTags.EBDigi = 'ecalDigis:ebDigis'
ecalDQMCollectionTags.EEDigi = 'ecalDigis:eeDigis'
ecalDQMCollectionTags.PnDiodeDigi = 'ecalDigis:'
ecalDQMCollectionTags.TrigPrimDigi = 'ecalDigis:EcalTriggerPrimitives'
ecalDQMCollectionTags.TrigPrimEmulDigi = 'valEcalTriggerPrimitiveDigis'

dqmInfoEcal = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Ecal')
)

ecalMonitorTask.tasks = cms.untracked.vstring(
    "OccupancyTask",
    "IntegrityTask",
    "RawDataTask",
    "PresampleTask",
    "TrigPrimTask",
    "ClusterTask",
    "EnergyTask",
    "TimingTask"
)
ecalMonitorTask.taskParameters.Common.hltTaskMode = 2
ecalMonitorTask.taskParameters.TrigPrimTask.runOnEmul = False
ecalMonitorTask.allowMissingCollections = True

## standard
ecal_dqm_source_offline = cms.Sequence(
    dqmInfoEcal *
    ecalMonitorTask *
    ecalBarrelHltTask *
    ecalBarrelClusterTaskExtras *
    ecalEndcapClusterTaskExtras *
    ecalzmasstask
)

## standard with Selective Readout Task
# ecalMonitorTaskSR = ecalMonitorTask.clone()
# ecalMonitorTaskSR.tasks = cms.untracked.vstring(
#     "OccupancyTask",
#     "IntegrityTask",
#     "RawDataTask",
#     "PresampleTask",
#     "TrigPrimTask",
#     "ClusterTask",
#     "EnergyTask",
#     "TimingTask",
#     "SelectiveReadoutTask"
# )

# ecal_dqm_source_offline1 = cms.Sequence(
#     dqmInfoEcal *
#     ecalMonitorTaskSR *
#     ecalBarrelClusterTaskExtras *
#     ecalEndcapClusterTaskExtras *
#     ecalzmasstask
# )

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
