import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalBarrelMonitorTasks.TowerStatusTask_cfi import ecalTowerStatusTask

ecalDaqInfoTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("Ecal DAQ Monitor"),
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

ecalDaqInfoTask.workerParameters.TowerStatusTask.params.doDCSInfo = False
