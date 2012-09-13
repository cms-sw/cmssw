import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.dqmpset import *
from DQM.EcalCommon.CommonParams_cfi import *

from DQM.EcalBarrelMonitorTasks.TowerStatusTask_cfi import ecalTowerStatusTask

from DQM.EcalCommon.EcalDQMBinningService_cfi import *

ecalBarrelDcsInfoTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("Ecal DCS Info"),
    mergeRuns = cms.untracked.bool(False),
    # tasks to be turned on
    workers = cms.untracked.vstring(
        "TowerStatusTask"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters = dqmpset(
        dict(
            TowerStatusTask = ecalTowerStatusTask,
            common = ecalCommonParams
        )
    ),
    collectionTags = cms.untracked.PSet(),
    allowMissingCollections = cms.untracked.bool(False),
    verbosity = cms.untracked.int32(0),
    evaluateTime = cms.untracked.bool(False)
)

ecalBarrelDcsInfoTask.workerParameters.TowerStatusTask.doDAQInfo = False
ecalBarrelDcsInfoTask.workerParameters.TowerStatusTask.doDCSInfo = True
