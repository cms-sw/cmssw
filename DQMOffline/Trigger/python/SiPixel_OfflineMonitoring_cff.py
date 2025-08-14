import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_Cluster_cff import *
from DQMOffline.Trigger.SiPixel_OfflineMonitoring_TrackCluster_cff import *
from RecoLocalTracker.SiPixelRecHits.SiPixelTemplateStoreESProducer_cfi import *

sipixelMonitorHLTsequence = cms.Sequence(
    hltSiPixelPhase1ClustersAnalyzer,
    #+ hltSiPixelPhase1TrackClustersAnalyzer,
    cms.Task(SiPixelTemplateStoreESProducer)
)
