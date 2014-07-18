import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalMonitorTasks.TowerStatusTask_cfi import ecalTowerStatusTask

ecalDcsInfoTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("Ecal DCS Monitor"),
    # tasks to be turned on
    workers = cms.untracked.vstring(
        "TowerStatusTask"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters =  cms.untracked.PSet(
        TowerStatusTask = ecalTowerStatusTask.clone()
    ),
    commonParameters = ecalCommonParams.clone(willConvertToEDM = cms.untracked.bool(False)),
    verbosity = cms.untracked.int32(0)
)

ecalDcsInfoTask.workerParameters.TowerStatusTask.params.doDAQInfo = False
