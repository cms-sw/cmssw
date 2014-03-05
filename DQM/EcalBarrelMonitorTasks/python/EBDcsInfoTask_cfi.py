import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalBarrelMonitorTasks.CollectionTags_cfi import ecalDQMCollectionTags

from DQM.EcalBarrelMonitorTasks.TowerStatusTask_cfi import ecalTowerStatusTask

ecalDcsInfoTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("Ecal Certification Monitor"),
    # tasks to be turned on
    workers = cms.untracked.vstring(
        "TowerStatusTask"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters =  cms.untracked.PSet(
        TowerStatusTask = ecalTowerStatusTask
    ),
    commonParameters = ecalCommonParams,
    collectionTags = ecalDQMCollectionTags,
    allowMissingCollections = cms.untracked.bool(True),
    verbosity = cms.untracked.int32(0),
    evaluateTime = cms.untracked.bool(False),
    resetInterval = cms.untracked.double(2.)
)

ecalDcsInfoTask.workerParameters.TowerStatusTask.params.doDAQInfo = False
ecalDcsInfoTask.workerParameters.TowerStatusTask.params.doDCSInfo = True

ecalBarrelDcsInfoTask = ecalDcsInfoTask.clone()
