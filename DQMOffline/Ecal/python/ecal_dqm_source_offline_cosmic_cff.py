import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import *
from DQM.EcalMonitorTasks.EcalFEDMonitor_cfi import *

dqmInfoEcal = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Ecal')
)

## standard
ecal_dqm_source_offline = cms.Sequence(
    dqmInfoEcal +
    ecalMonitorTask +
    ecalFEDMonitor
)

ecalMonitorTask.workerParameters.TrigPrimTask.params.runOnEmul = False
ecalMonitorTask.collectionTags.EBBasicCluster = 'cosmicBasicClusters:CosmicBarrelBasicClusters'
ecalMonitorTask.collectionTags.EEBasicCluster = 'cosmicBasicClusters:CosmicEndcapBasicClusters'
ecalMonitorTask.collectionTags.EBSuperCluster = 'cosmicSuperClusters:CosmicBarrelSuperClusters'
ecalMonitorTask.collectionTags.EESuperCluster = 'cosmicSuperClusters:CosmicEndcapSuperClusters'
