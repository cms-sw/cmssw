import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalMonitorClient.TowerStatusTask_cfi import ecalTowerStatusTask

ecalDcsInfoTask = DQMEDHarvester("EcalDQMonitorClient",
    moduleName = cms.untracked.string("Ecal DCS Monitor"),
    # tasks to be turned on
    workers = cms.untracked.vstring(
        "TowerStatusTask"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters =  cms.untracked.PSet(
        TowerStatusTask = ecalTowerStatusTask.clone()
    ),
    commonParameters = ecalCommonParams,
    verbosity = cms.untracked.int32(0)
)

ecalDcsInfoTask.workerParameters.TowerStatusTask.params.doDAQInfo = False
