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
from DQM.EcalMonitorTasks.ClusterTask_cfi import ecalClusterTaskPhase2
from DQM.EcalMonitorTasks.EnergyTask_cfi import ecalEnergyTaskPhase2
from DQM.EcalMonitorTasks.TimingTask_cfi import ecalTimingTaskPhase2
from DQM.EcalMonitorTasks.ecalPiZeroTask_cfi import ecalPiZeroTask

ecalMonitorTaskEcalOnlyPhase2 = ecalMonitorTaskEcalOnly.clone(
    workers = cms.untracked.vstring(
        "ClusterTask",
        "EnergyTask",
        "TimingTask",
        "PiZeroTask"
    ),
    workerParameters = cms.untracked.PSet(
        ClusterTask = ecalClusterTaskPhase2,
        EnergyTask = ecalEnergyTaskPhase2,
        TimingTask = ecalTimingTaskPhase2,
        PiZeroTask = ecalPiZeroTask
    ),
    collectionTags = ecalDQMCollectionTagsPhase2,
    # skip EcalRawData collection to prevent event type filtering while no Phase 2 raw data is defined
    skipCollections = cms.untracked.vstring('EcalRawData')
)
