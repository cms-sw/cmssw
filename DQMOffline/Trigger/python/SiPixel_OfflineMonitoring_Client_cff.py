import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_Cluster_cff import *
from DQMOffline.Trigger.SiPixel_OfflineMonitoring_TrackCluster_cff import *

from DQM.HLTEvF.dqmCorrelationClient_cfi import *
pixelClusterVsLumi = dqmCorrelationClient.clone(
   me = cms.PSet(
      folder = cms.string("HLT/Pixel/"),
      name   = cms.string("num_clusters_per_instLumi"),
      doXaxis = cms.bool( True ),
      nbinsX = cms.int32(    40 ),
      xminX  = cms.double(    0.),
      xmaxX  = cms.double(20000.),
#      doYaxis = cms.bool( False ),
      doYaxis = cms.bool( True ),
      nbinsY = cms.int32 (    400 ),
      xminY  = cms.double(      0.),
      xmaxY  = cms.double( 400000.),
   ),
   me1 = cms.PSet(
      folder   = cms.string("HLT/LumiMonitoring"),
      name     = cms.string("lumiVsLS"),
      profileX = cms.bool(True)
   ),
   me2 = cms.PSet(
      folder   = cms.string("HLT/Pixel"),
      name     = cms.string("num_clusters_per_Lumisection_PXBarrel"),
      profileX = cms.bool(True)
   ),
)

pixelClusterVsLumiPXBarrel = pixelClusterVsLumi.clone()
pixelClusterVsLumiPXBarrel.me.name  = "num_clusters_per_instLumi_PXBarrel" 
pixelClusterVsLumiPXBarrel.me2.name = "num_clusters_per_Lumisection_PXBarrel"

pixelClusterVsLumiPXForward = pixelClusterVsLumi.clone()
pixelClusterVsLumiPXForward.me.name  = "num_clusters_per_instLumi_PXForward" 
pixelClusterVsLumiPXForward.me2.name = "num_clusters_per_Lumisection_PXForward"

pixelTrackClusterVsLumiPXBarrel = pixelClusterVsLumi.clone()
pixelTrackClusterVsLumiPXBarrel.me.folder  = "HLT/Pixel/TrackClusters/"
pixelTrackClusterVsLumiPXBarrel.me.name    = "num_clusters_ontrack_per_instLumi_PXBarrel" 
pixelTrackClusterVsLumiPXBarrel.me2.folder = "HLT/Pixel/TrackClusters"
pixelTrackClusterVsLumiPXBarrel.me2.name   = "num_clusters_ontrack_per_Lumisection_PXBarrel"

pixelTrackClusterVsLumiPXForward = pixelClusterVsLumi.clone()
pixelTrackClusterVsLumiPXForward.me.folder  = "HLT/Pixel/TrackClusters/"
pixelTrackClusterVsLumiPXForward.me.name    = "num_clusters_ontrack_per_instLumi_PXForward" 
pixelTrackClusterVsLumiPXForward.me2.folder = "HLT/Pixel/TrackClusters"
pixelTrackClusterVsLumiPXForward.me2.name   = "num_clusters_ontrack_per_Lumisection_PXForward"

sipixelHarvesterHLTsequence = cms.Sequence(
#    hltSiPixelPhase1ClustersHarvester
#    + hltSiPixelPhase1TrackClustersHarvester
    pixelClusterVsLumiPXBarrel
    + pixelClusterVsLumiPXForward
#    + pixelTrackClusterVsLumiPXBarrel
#    + pixelTrackClusterVsLumiPXForward
)    
