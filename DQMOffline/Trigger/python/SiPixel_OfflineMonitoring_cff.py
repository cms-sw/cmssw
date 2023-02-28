import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_Cluster_cff import *
from DQMOffline.Trigger.SiPixel_OfflineMonitoring_TrackCluster_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
from DQM.SiPixelMonitorTrack.RefitterForPixelDQM import *

hltSiPixelClusterShapeCache = siPixelClusterShapeCache.clone(src = 'hltSiPixelClusters')
hltrefittedForPixelDQM = refittedForPixelDQM.clone(src ='hltMergedTracks',
                                                   TTRHBuilder = 'WithTrackAngle') # no templates at HLT
 

sipixelMonitorHLTsequence = cms.Sequence(
    hltSiPixelClusterShapeCache
    + hltSiPixelPhase1ClustersAnalyzer
    + hltrefittedForPixelDQM
    + hltSiPixelPhase1TrackClustersAnalyzer
)    
