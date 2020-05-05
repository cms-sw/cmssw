import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import ecalMonitorTask as _ecalMonitorTask

ecalMonitorTaskEcalOnly = _ecalMonitorTask.clone()
ecalMonitorTaskEcalOnly.collectionTags.EBSuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECALHLT","particleFlowSuperClusterECALBarrel")
ecalMonitorTaskEcalOnly.collectionTags.EESuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECALHLT","particleFlowSuperClusterECALEndcapWithPreshower")
ecalMonitorTaskEcalOnly.workerParameters.ClusterTask.params.doExtra = cms.untracked.bool(False)
