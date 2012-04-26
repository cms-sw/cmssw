import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.dqmpset import *
from DQM.EcalCommon.CollectionTags_cfi import *
from DQM.EcalCommon.CommonParams_cfi import *

import DQM.EcalBarrelMonitorTasks.ClusterTask_cfi as ecalClusterTask
import DQM.EcalBarrelMonitorTasks.EnergyTask_cfi as ecalEnergyTask
import DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi as ecalIntegrityTask
import DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi as ecalOccupancyTask
import DQM.EcalBarrelMonitorTasks.RawDataTask_cfi as ecalRawDataTask
import DQM.EcalBarrelMonitorTasks.SelectiveReadoutTask_cfi as ecalSelectiveReadoutTask
import DQM.EcalBarrelMonitorTasks.TimingTask_cfi as ecalTimingTask
import DQM.EcalBarrelMonitorTasks.TrigPrimTask_cfi as ecalTrigPrimTask
import DQM.EcalBarrelMonitorTasks.TowerStatusTask_cfi as ecalTowerStatusTask
import DQM.EcalBarrelMonitorTasks.PresampleTask_cfi as ecalPresampleTask

ecalMonitorTaskParams = dict(
    ClusterTask = ecalClusterTask.clusterTask,
    EnergyTask = ecalEnergyTask.energyTask,
    IntegrityTask = ecalIntegrityTask.integrityTask,
    OccupancyTask = ecalOccupancyTask.occupancyTask,
    RawDataTask = ecalRawDataTask.rawDataTask,
    SelectiveReadoutTask = ecalSelectiveReadoutTask.selectiveReadoutTask,
    TimingTask = ecalTimingTask.timingTask,
    TrigPrimTask = ecalTrigPrimTask.trigPrimTask,
    TowerStatusTask = ecalTowerStatusTask.towerStatusTask,
    PresampleTask = ecalPresampleTask.presampleTask,
    Common = ecalCommonParams
)
        
ecalMonitorTaskPaths = dict(
    ClusterTask = ecalClusterTask.clusterTaskPaths,
    EnergyTask = ecalEnergyTask.energyTaskPaths,
    IntegrityTask = ecalIntegrityTask.integrityTaskPaths,
    OccupancyTask = ecalOccupancyTask.occupancyTaskPaths,
    RawDataTask = ecalRawDataTask.rawDataTaskPaths,
    SelectiveReadoutTask = ecalSelectiveReadoutTask.selectiveReadoutTaskPaths,
    TimingTask = ecalTimingTask.timingTaskPaths,
    TrigPrimTask = ecalTrigPrimTask.trigPrimTaskPaths,
    TowerStatusTask = ecalTowerStatusTask.towerStatusTaskPaths,
    PresampleTask = ecalPresampleTask.presampleTaskPaths
)

ecalMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("Ecal Monitor Source"),
    # tasks to be turned on
    tasks = cms.untracked.vstring(
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
    taskParameters = dqmpset(ecalMonitorTaskParams),
    # ME paths for each task (included from inidividual cfis)
    mePaths = dqmpaths("Ecal", ecalMonitorTaskPaths),
    collectionTags = ecalDQMCollectionTags,
    allowMissingCollections = cms.untracked.bool(False),
    verbosity = cms.untracked.int32(0)
)

