import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingTrk
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer as _ticlLayerTileProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersTrk = _filteredLayerClustersProducer.clone(
  clusterFilter = "ClusterFilterByAlgo",
  algo_number = 8,
  LayerClustersInputMask = 'trackstersEM',
  iteration_label = "Trk"
)

# CA - PATTERN RECOGNITION

trackstersTrk = _trackstersProducer.clone(
  filtered_mask = cms.InputTag("filteredLayerClustersTrk", "Trk"),
  original_mask = 'trackstersEM',
  seeding_regions = "ticlSeedingTrk",
  filter_on_categories = [2, 4], # filter muons and charged hadrons
  pid_threshold = 0.0,
  missing_layers = 5,
  min_clusters_per_ntuplet = 10,
  min_cos_theta = 0.978, # ~12 degrees
  min_cos_pointing = 0.866,
  max_delta_time = -1.
)

# MULTICLUSTERS

multiClustersFromTrackstersTrk = _multiClustersFromTrackstersProducer.clone(
    label = "TrkMultiClustersFromTracksterByCA",
    Tracksters = "trackstersTrk"
)

TrkStepTask = cms.Task(ticlSeedingTrk
    ,filteredLayerClustersTrk
    ,trackstersTrk
    ,multiClustersFromTrackstersTrk)

