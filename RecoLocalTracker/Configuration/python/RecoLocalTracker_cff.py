import FWCore.ParameterSet.Config as cms

#
# Tracker Local Reco
# Initialize magnetic field
#
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import *
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
from RecoLocalTracker.SubCollectionProducers.clustersummaryproducer_cfi import *

pixeltrackerlocalreco = cms.Task(siPixelClustersPreSplitting,siPixelRecHitsPreSplitting)
striptrackerlocalreco = cms.Task(siStripZeroSuppression,siStripClusters,siStripMatchedRecHits)
trackerlocalreco = cms.Task(pixeltrackerlocalreco,striptrackerlocalreco,clusterSummaryProducer)

from RecoLocalTracker.SiPhase2Clusterizer.phase2TrackerClusterizer_cfi import *
from RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEGeometricESProducer_cfi import *

_pixeltrackerlocalreco_phase2 = pixeltrackerlocalreco.copy()
_pixeltrackerlocalreco_phase2.add(siPhase2Clusters)
phase2_tracker.toReplaceWith(pixeltrackerlocalreco, _pixeltrackerlocalreco_phase2)
phase2_tracker.toReplaceWith(trackerlocalreco, trackerlocalreco.copyAndExclude([striptrackerlocalreco]))
