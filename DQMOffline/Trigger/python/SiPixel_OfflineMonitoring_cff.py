import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_Cluster_cff import *
from DQMOffline.Trigger.SiPixel_OfflineMonitoring_TrackCluster_cff import *
from DQMOffline.Trigger.SiPixel_OfflineMonitoring_TrackResiduals_cff import *
from RecoTracker.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
from DQM.SiPixelMonitorTrack.RefitterForPixelDQM import *
from RecoLocalTracker.SiPixelRecHits.SiPixelTemplateStoreESProducer_cfi import *

hltSiPixelClusterShapeCacheForDQM = siPixelClusterShapeCache.clone(src = 'hltSiPixelClusters')

from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
pp_on_PbPb_run3.toModify(hltSiPixelClusterShapeCacheForDQM,
                         src =  "hltSiPixelClustersAfterSplittingPPOnAA")

hltTrackRefitterForPixelDQM = refittedForPixelDQM.clone(src                     = "hltMergedTracks",
                                                        beamSpot                = "hltOnlineBeamSpot",
                                                        TTRHBuilder             = 'hltESPTTRHBWithTrackAngle') # no template at HLT

pp_on_PbPb_run3.toModify(hltTrackRefitterForPixelDQM,
                         src ='hltMergedTracksPPOnAA')

sipixelMonitorHLTsequence = cms.Sequence(
    hltSiPixelClusterShapeCacheForDQM
    + hltSiPixelPhase1ClustersAnalyzer
    + hltTrackRefitterForPixelDQM
    + hltSiPixelPhase1TrackClustersAnalyzer
    + hltSiPixelPhase1TrackResidualsAnalyzer,
    cms.Task(SiPixelTemplateStoreESProducer)
)
