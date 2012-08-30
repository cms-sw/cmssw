import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.dqmpset import *
from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalBarrelMonitorTasks.CollectionTags_cfi import ecalDQMCollectionTags

from DQM.EcalBarrelMonitorTasks.ClusterTask_cfi import clusterTask
from DQM.EcalBarrelMonitorTasks.EnergyTask_cfi import energyTask
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import integrityTask
from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import occupancyTask
from DQM.EcalBarrelMonitorTasks.RawDataTask_cfi import rawDataTask
from DQM.EcalBarrelMonitorTasks.SelectiveReadoutTask_cfi import selectiveReadoutTask
from DQM.EcalBarrelMonitorTasks.TimingTask_cfi import timingTask
from DQM.EcalBarrelMonitorTasks.TrigPrimTask_cfi import trigPrimTask
from DQM.EcalBarrelMonitorTasks.TowerStatusTask_cfi import towerStatusTask
from DQM.EcalBarrelMonitorTasks.PresampleTask_cfi import presampleTask

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
        "SelectiveReadoutTask",
        "TimingTask",
        "TrigPrimTask",
        "TowerStatusTask",
        "PresampleTask"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters = dqmpset(
        dict(
            ClusterTask = clusterTask,
            EnergyTask = energyTask,
            IntegrityTask = integrityTask,
            OccupancyTask = occupancyTask,
            RawDataTask = rawDataTask,
            SelectiveReadoutTask = selectiveReadoutTask,
            TimingTask = timingTask,
            TrigPrimTask = trigPrimTask,
            TowerStatusTask = towerStatusTask,
            PresampleTask = presampleTask,
            common = ecalCommonParams
        )
    ),
    collectionTags = ecalDQMCollectionTags,
    allowMissingCollections = cms.untracked.bool(False),
    verbosity = cms.untracked.int32(0),
    evaluateTime = cms.untracked.bool(False)
)
