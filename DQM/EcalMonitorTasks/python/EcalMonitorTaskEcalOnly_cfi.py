import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import ecalMonitorTask as _ecalMonitorTask

ecalMonitorTaskEcalOnly = _ecalMonitorTask.clone(
    collectionTags = dict(
        EBSuperCluster = ("particleFlowSuperClusterECALOnly","particleFlowSuperClusterECALBarrel"),
        EESuperCluster = ("particleFlowSuperClusterECALOnly","particleFlowSuperClusterECALEndcapWithPreshower")
    ),
    workerParameters = dict(ClusterTask = dict(params = dict(doExtra = False)))
)

# Changes for Phase 2
from DQM.EcalMonitorTasks.CollectionTags_cfi import ecalDQMCollectionTagsPhase2
from DQM.EcalMonitorTasks.ClusterTask_cfi import ecalClusterTask
from DQM.EcalMonitorTasks.EnergyTask_cfi import ecalEnergyTask
from DQM.EcalMonitorTasks.TimingTask_cfi import ecalTimingTask
from DQM.EcalMonitorTasks.ecalPiZeroTask_cfi import ecalPiZeroTask

ecalMonitorTaskEcalOnlyPhase2 = ecalMonitorTaskEcalOnly.clone(
    workers = cms.untracked.vstring(
        "ClusterTask",
        "EnergyTask",
        "TimingTask",
        "PiZeroTask"
    ),
    workerParameters = cms.untracked.PSet(
        ClusterTask = ecalClusterTask,
        EnergyTask = ecalEnergyTask,
        TimingTask = ecalTimingTask,
        PiZeroTask = ecalPiZeroTask
    ),
    collectionTags = ecalDQMCollectionTagsPhase2,
    skipCollections = cms.untracked.vstring('EcalRawData')
)
