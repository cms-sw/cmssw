import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import ecalMonitorTask as _ecalMonitorTask

ecalMonitorTaskEcalOnly = _ecalMonitorTask.clone()
ecalMonitorTaskEcalOnly.collectionTags.EBSuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECALOnly","particleFlowSuperClusterECALBarrel")
ecalMonitorTaskEcalOnly.collectionTags.EESuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECALOnly","particleFlowSuperClusterECALEndcapWithPreshower")
ecalMonitorTaskEcalOnly.workerParameters.ClusterTask.params.doExtra = cms.untracked.bool(False)
