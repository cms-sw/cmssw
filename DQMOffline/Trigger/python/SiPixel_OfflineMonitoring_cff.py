import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_Cluster_cff import *
from DQMOffline.Trigger.SiPixel_OfflineMonitoring_TrackCluster_cff import *
from RecoTracker.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
from DQM.SiPixelMonitorTrack.RefitterForPixelDQM import *
from RecoLocalTracker.SiPixelRecHits.SiPixelTemplateStoreESProducer_cfi import *

hltSiPixelClusterShapeCache = siPixelClusterShapeCache.clone(src = 'hltSiPixelClusters')
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(hltSiPixelClusterShapeCache, src = cms.InputTag("siPixelClusters","","HLT"))

hltrefittedForPixelDQM = refittedForPixelDQM.clone(src ='hltMergedTracks',
                                                   TTRHBuilder = 'WithTrackAngle') # no templates at HLT
phase2_tracker.toModify(hltrefittedForPixelDQM, src = cms.InputTag("generalTracks","","HLT"))

sipixelMonitorHLTsequence = cms.Sequence(
    hltSiPixelClusterShapeCache
    + hltSiPixelPhase1ClustersAnalyzer
    + hltrefittedForPixelDQM
    + hltSiPixelPhase1TrackClustersAnalyzer,
    cms.Task(SiPixelTemplateStoreESProducer)
)    
