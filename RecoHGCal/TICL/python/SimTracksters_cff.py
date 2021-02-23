import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.trackstersFromSimClustersProducer_cfi import trackstersFromSimClustersProducer as _trackstersFromSimClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer


# CA - PATTERN RECOGNITION


filteredLayerClustersSimTracksters = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    algo_number = 8,
    min_cluster_size = 0, # inclusive
    iteration_label = "ticlSimTracksters"
)

ticlSimTracksters = _trackstersFromSimClustersProducer.clone(
)

ticlMultiClustersFromSimTracksters = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "ticlSimTracksters"
)

ticlSimTrackstersTask = cms.Task(filteredLayerClustersSimTracksters, ticlSimTracksters, ticlMultiClustersFromSimTracksters)

