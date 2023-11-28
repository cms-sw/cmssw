import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_Cluster_cff import *
from DQMOffline.Trigger.SiPixel_OfflineMonitoring_TrackCluster_cff import *
<<<<<<< HEAD
from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
=======
from RecoTracker.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
>>>>>>> 895df58e36cff1d7dc27b1bf37aee7f604adc704
from DQM.SiPixelMonitorTrack.RefitterForPixelDQM import *
from RecoLocalTracker.SiPixelRecHits.SiPixelTemplateStoreESProducer_cfi import *

hltSiPixelClusterShapeCache = siPixelClusterShapeCache.clone(src = 'hltSiPixelClusters')
<<<<<<< HEAD
hltrefittedForPixelDQM = refittedForPixelDQM.clone(src ='hltMergedTracks',
                                                   TTRHBuilder = 'WithTrackAngle') # no templates at HLT
 
=======

from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
pp_on_PbPb_run3.toModify(hltSiPixelClusterShapeCache,
                         src =  "hltSiPixelClustersAfterSplittingPPOnAA")

hltrefittedForPixelDQM = refittedForPixelDQM.clone(src ='hltMergedTracks',
                                                   TTRHBuilder = 'WithTrackAngle') # no templates at HLT

pp_on_PbPb_run3.toModify(hltrefittedForPixelDQM,
                         src ='hltMergedTracksPPOnAA')
>>>>>>> 895df58e36cff1d7dc27b1bf37aee7f604adc704

sipixelMonitorHLTsequence = cms.Sequence(
    hltSiPixelClusterShapeCache
    + hltSiPixelPhase1ClustersAnalyzer
    + hltrefittedForPixelDQM
    + hltSiPixelPhase1TrackClustersAnalyzer,
    cms.Task(SiPixelTemplateStoreESProducer)
<<<<<<< HEAD
)    
=======
)
>>>>>>> 895df58e36cff1d7dc27b1bf37aee7f604adc704
