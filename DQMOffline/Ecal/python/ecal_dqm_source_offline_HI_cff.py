import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import *
from DQM.EcalMonitorTasks.EcalFEDMonitor_cfi import *
from DQMOffline.Ecal.EcalZmassTask_cfi import *

dqmInfoEcal = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Ecal')
)

## standard
ecal_dqm_source_offline = cms.Sequence(
    dqmInfoEcal +
    ecalMonitorTask +
    ecalFEDMonitor +
    ecalzmasstask
)

ecalMonitorTask.collectionTags.EBBasicCluster = 'islandBasicClusters:islandBarrelBasicClusters'
ecalMonitorTask.collectionTags.EEBasicCluster = 'islandBasicClusters:islandEndcapBasicClusters'
ecalMonitorTask.collectionTags.EBSuperCluster = 'correctedIslandBarrelSuperClusters'
ecalMonitorTask.collectionTags.EESuperCluster = 'correctedIslandEndcapSuperClusters'
