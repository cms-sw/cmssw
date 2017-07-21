import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_Cluster_cff import *
from DQMOffline.Trigger.SiPixel_OfflineMonitoring_TrackCluster_cff import *

sipixelMonitorHLTsequence = cms.Sequence(
    hltSiPixelPhase1ClustersAnalyzer
    + hltSiPixelPhase1TrackClustersAnalyzer
)    
