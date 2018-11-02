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

pixeltrackerlocalreco = cms.Sequence(siPixelClustersPreSplitting*siPixelRecHitsPreSplitting)
striptrackerlocalreco = cms.Sequence(siStripZeroSuppression*siStripClusters*siStripMatchedRecHits)
trackerlocalreco = cms.Sequence(pixeltrackerlocalreco*striptrackerlocalreco*clusterSummaryProducer)

from RecoLocalTracker.SiPhase2Clusterizer.phase2TrackerClusterizer_cfi import *
from RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEGeometricESProducer_cfi import *

phase2_tracker.toReplaceWith(pixeltrackerlocalreco,
  cms.Sequence(
          siPhase2Clusters +
          siPixelClustersPreSplitting +
          siPixelRecHitsPreSplitting
  )
)
phase2_tracker.toReplaceWith(trackerlocalreco,
  cms.Sequence(
          pixeltrackerlocalreco*clusterSummaryProducer
  )
)
