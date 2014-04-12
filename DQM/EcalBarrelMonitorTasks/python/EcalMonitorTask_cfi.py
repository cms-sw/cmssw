import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalBarrelMonitorTasks.CollectionTags_cfi import ecalDQMCollectionTags

from DQM.EcalBarrelMonitorTasks.ClusterTask_cfi import ecalClusterTask
from DQM.EcalBarrelMonitorTasks.EnergyTask_cfi import ecalEnergyTask
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask
from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalBarrelMonitorTasks.PresampleTask_cfi import ecalPresampleTask
from DQM.EcalBarrelMonitorTasks.RawDataTask_cfi import ecalRawDataTask
from DQM.EcalBarrelMonitorTasks.RecoSummaryTask_cfi import ecalRecoSummaryTask
from DQM.EcalBarrelMonitorTasks.SelectiveReadoutTask_cfi import ecalSelectiveReadoutTask
from DQM.EcalBarrelMonitorTasks.TimingTask_cfi import ecalTimingTask
from DQM.EcalBarrelMonitorTasks.TrigPrimTask_cfi import ecalTrigPrimTask

ecalMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("Ecal Monitor Source"),
    # tasks to be turned on
    workers = cms.untracked.vstring(
        "ClusterTask",
        "EnergyTask",
        "IntegrityTask",
        "OccupancyTask",
        "PresampleTask",
        "RawDataTask",
        "RecoSummaryTask",
        "TrigPrimTask"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters =  cms.untracked.PSet(
        ClusterTask = ecalClusterTask,
        EnergyTask = ecalEnergyTask,
        IntegrityTask = ecalIntegrityTask,
        OccupancyTask = ecalOccupancyTask,
        PresampleTask = ecalPresampleTask,
        RawDataTask = ecalRawDataTask,
        RecoSummaryTask = ecalRecoSummaryTask,
        SelectiveReadoutTask = ecalSelectiveReadoutTask,
        TimingTask = ecalTimingTask,
        TrigPrimTask = ecalTrigPrimTask
    ),
    commonParameters = ecalCommonParams,
    collectionTags = ecalDQMCollectionTags,
    allowMissingCollections = cms.untracked.bool(True),
    verbosity = cms.untracked.int32(0),
    evaluateTime = cms.untracked.bool(False),
    resetInterval = cms.untracked.double(2.)
)



