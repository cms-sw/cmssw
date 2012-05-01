import FWCore.ParameterSet.Config as cms

ecalBarrelDcsInfoTask = cms.EDAnalyzer("EBDcsInfoTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False)
)

# from DQM.EcalCommon.dqmpset import *
# from DQM.EcalCommon.CollectionTags_cfi import *
# from DQM.EcalCommon.CommonParams_cfi import *

# from DQM.EcalCommon.EcalDQMBinningService_cfi import *

# import DQM.EcalBarrelMonitorTasks.TowerStatusTask_cfi as ecalTowerStatusTask

# ecalMonitorTaskParams = dict(
#     TowerStatusTask = ecalTowerStatusTask.towerStatusTask,
#     Common = ecalCommonParams
# )

# ecalMonitorTaskPaths = dict(
#     TowerStatusTask = ecalTowerStatusTask.towerStatusTaskPaths
# )

# ecalBarrelDcsInfoTask = cms.EDAnalyzer("EcalDQMonitorTask",
#     moduleName = cms.untracked.string("Ecal DCS Info"),
#     # tasks to be turned on
#     tasks = cms.untracked.vstring(
#         "TowerStatusTask"
#     ),
#     # task parameters (included from indivitual cfis)
#     taskParameters = dqmpset(ecalMonitorTaskParams),
#     # ME paths for each task (included from inidividual cfis)
#     mePaths = dqmpaths("Ecal", ecalMonitorTaskPaths),
#     collectionTags = ecalDQMCollectionTags,
#     allowMissingCollections = cms.untracked.bool(False),
#     verbosity = cms.untracked.int32(0)
# )

# ecalBarrelDcsInfoTask.taskParameters.TowerStatusTask.doDAQInfo = False
# ecalBarrelDcsInfoTask.taskParameters.TowerStatusTask.doDAQInfo = True
