import FWCore.ParameterSet.Config as cms
from DQM.SiTrackerPhase2.Phase2ITMonitorCluster_cff import *
from DQM.SiTrackerPhase2.Phase2OTMonitorCluster_cff import *

HLTclusterMonitorIT = clusterMonitorIT.clone(
    TopFolderName = cms.string('HLT/TrackerPhase2ITCluster'),
    InnerPixelClusterSource = cms.InputTag('siPixelClusters','','HLT'),
)
HLTclusterMonitorOT = clusterMonitorOT.clone(
    TopFolderName = cms.string('HLT/TrackerPhase2OTCluster'),
    clusterSrc = cms.InputTag('siPhase2Clusters','','HLT'),
)

HLTtrackerphase2DQMSource = cms.Sequence(HLTclusterMonitorIT +
                                         HLTclusterMonitorOT)

