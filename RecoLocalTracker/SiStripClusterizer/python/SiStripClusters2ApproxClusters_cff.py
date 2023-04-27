from RecoLocalTracker.SiStripClusterizer.SiStripClusters2ApproxClusters_cfi import *

from Configuration.ProcessModifiers.approxSiStripClusters_cff import approxSiStripClusters

from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import ClusterShapeHitFilterESProducer as _ClusterShapeHitFilterESProducer
hltClusterShapeHitFilterESProducer = _ClusterShapeHitFilterESProducer.clone(ComponentName = 'hltClusterShapeHitFilterESProducer')

from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import *
hltSiStripClusters2ApproxClusters = SiStripClusters2ApproxClusters.clone()
approxSiStripClusters.toModify(hltSiStripClusters2ApproxClusters,
                               beamSpot = "onlineBeamSpotProducer",
                               inputClusters = "siStripClustersHLT",
                               clusterShapeHitFilterLabel = "hltClusterShapeHitFilterESProducer")
