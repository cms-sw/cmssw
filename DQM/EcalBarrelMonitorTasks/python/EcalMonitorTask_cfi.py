import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalBarrelMonitorTasks.CollectionTags_cfi import ecalDQMCollectionTags

from DQM.EcalBarrelMonitorTasks.ClusterTask_cfi import ecalClusterTask
from DQM.EcalBarrelMonitorTasks.EnergyTask_cfi import ecalEnergyTask
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask
from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalBarrelMonitorTasks.RawDataTask_cfi import ecalRawDataTask
from DQM.EcalBarrelMonitorTasks.SelectiveReadoutTask_cfi import ecalSelectiveReadoutTask
from DQM.EcalBarrelMonitorTasks.TimingTask_cfi import ecalTimingTask
from DQM.EcalBarrelMonitorTasks.TrigPrimTask_cfi import ecalTrigPrimTask
from DQM.EcalBarrelMonitorTasks.TowerStatusTask_cfi import ecalTowerStatusTask
from DQM.EcalBarrelMonitorTasks.PresampleTask_cfi import ecalPresampleTask

ecalMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("Ecal Monitor Source"),
    mergeRuns = cms.untracked.bool(False),
    # tasks to be turned on
    workers = cms.untracked.vstring(
        "ClusterTask",
        "EnergyTask",
        "IntegrityTask",
        "OccupancyTask",
        "RawDataTask",
        "TimingTask",
        "TrigPrimTask",
        "TowerStatusTask",
        "PresampleTask"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters =  cms.untracked.PSet(
        ClusterTask = ecalClusterTask,
        EnergyTask = ecalEnergyTask,
        IntegrityTask = ecalIntegrityTask,
        OccupancyTask = ecalOccupancyTask,
        RawDataTask = ecalRawDataTask,
        SelectiveReadoutTask = ecalSelectiveReadoutTask,
        TimingTask = ecalTimingTask,
        TrigPrimTask = ecalTrigPrimTask,
        TowerStatusTask = ecalTowerStatusTask,
        PresampleTask = ecalPresampleTask,
        common = ecalCommonParams
    ),
    collectionTags = ecalDQMCollectionTags,
    allowMissingCollections = cms.untracked.bool(True),
    verbosity = cms.untracked.int32(0),
    evaluateTime = cms.untracked.bool(False),
    online = cms.untracked.bool(False),
    resetInterval = cms.untracked.double(2.)
)
