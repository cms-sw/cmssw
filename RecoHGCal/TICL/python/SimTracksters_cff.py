import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.trackstersFromSimClustersProducer_cfi import trackstersFromSimClustersProducer as _trackstersFromSimClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer


# CA - PATTERN RECOGNITION

ticlSimTracksters = _trackstersFromSimClustersProducer.clone(
)

ticlMultiClustersFromSimTracksters = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "ticlSimTracksters"
)

ticlSimTrackstersTask = cms.Task(ticlSimTracksters, ticlMultiClustersFromSimTracksters)

