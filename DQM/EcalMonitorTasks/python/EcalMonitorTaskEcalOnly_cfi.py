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
from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
from DQM.EcalMonitorTasks.CollectionTags_cfi import ecalDQMCollectionTagsPhase2
from DQM.EcalMonitorTasks.EnergyTask_cfi import ecalEnergyTask
from DQM.EcalMonitorTasks.TimingTask_cfi import ecalTimingTask
from DQM.EcalMonitorTasks.ecalPiZeroTask_cfi import ecalPiZeroTask

ecalMonitorTaskEcalOnlyPhase2 = ecalMonitorTaskEcalOnly.clone(
    workers = cms.untracked.vstring(
        "EnergyTask",
        "TimingTask",
        "PiZeroTask"
    ),
    workerParameters = cms.untracked.PSet(
        EnergyTask = ecalEnergyTask,
        TimingTask = ecalTimingTask,
        PiZeroTask = ecalPiZeroTask
    ),
    collectionTags = ecalDQMCollectionTagsPhase2,
    skipCollections = cms.untracked.vstring('EcalRawData')
)
