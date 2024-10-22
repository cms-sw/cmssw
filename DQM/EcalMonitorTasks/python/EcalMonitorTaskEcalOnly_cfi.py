import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import ecalMonitorTask as _ecalMonitorTask

ecalMonitorTaskEcalOnly = _ecalMonitorTask.clone(
       collectionTags = dict(
               EBSuperCluster = ("particleFlowSuperClusterECALOnly","particleFlowSuperClusterECALBarrel"),
               EESuperCluster = ("particleFlowSuperClusterECALOnly","particleFlowSuperClusterECALEndcapWithPreshower")
       ),
      workerParameters = dict(ClusterTask = dict(params = dict(doExtra = False)))   
)
