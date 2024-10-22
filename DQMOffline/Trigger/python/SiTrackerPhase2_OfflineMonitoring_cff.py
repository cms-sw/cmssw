import FWCore.ParameterSet.Config as cms
from DQM.SiTrackerPhase2.Phase2ITMonitorCluster_cff import *
from DQM.SiTrackerPhase2.Phase2OTMonitorCluster_cff import *

HLTclusterMonitorIT = clusterMonitorIT.clone(
    TopFolderName = cms.string('HLT/TrackerPhase2ITCluster'),
    InnerPixelClusterSource = cms.InputTag('hltSiPixelClusters'),
)
HLTclusterMonitorOT = clusterMonitorOT.clone(
    TopFolderName = cms.string('HLT/TrackerPhase2OTCluster'),
    clusterSrc = cms.InputTag('hltSiPhase2Clusters'),
)

HLTtrackerphase2DQMSource = cms.Sequence(HLTclusterMonitorIT +
                                         HLTclusterMonitorOT)

