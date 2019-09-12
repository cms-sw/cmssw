import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.ticlSeedingRegionProducer_cfi import ticlSeedingRegionProducer as _ticlSeedingRegionProducer
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer as _ticlLayerTileProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# SEEDING REGION

ticlSeedingTrk = _ticlSeedingRegionProducer.clone(
  algoId = 1
)

# CLUSTER FILTERING/MASKING

filteredLayerClustersTrk = _filteredLayerClustersProducer.clone(
  clusterFilter = "ClusterFilterByAlgo",
  algo_number = 8,
  iteration_label = "Trk"
)

# CA - PATTERN RECOGNITION

trackstersTrk = _trackstersProducer.clone(
  filtered_mask = cms.InputTag("filteredLayerClustersTrk", "Trk"),
  seeding_regions = "ticlSeedingTrk",
  missing_layers = 3,
  min_clusters_per_ntuplet = 5,
  min_cos_theta = 0.99, # ~10 degrees
  min_cos_pointing = 0.9
)

# MULTICLUSTERS

multiClustersFromTrackstersTrk = _multiClustersFromTrackstersProducer.clone(
    label = "TrkMultiClustersFromTracksterByCA",
    Tracksters = "trackstersTrk"
)

TrkStepTask = cms.Task(ticlSeedingTrk,
    filteredLayerClustersTrk,
    trackstersTrk,
    multiClustersFromTrackstersTrk)

TrkStep = cms.Sequence(TrkStepTask)
