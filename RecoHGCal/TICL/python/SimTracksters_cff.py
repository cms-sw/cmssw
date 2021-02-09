import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.trackstersFromSimClustersProducer_cfi import trackstersFromSimClustersProducer as _trackstersFromSimClustersProducer


# CA - PATTERN RECOGNITION

ticlSimTracksters = _trackstersFromSimClustersProducer.clone(
)


ticlSimTrackstersTask = cms.Task(ticlSimTracksters,)

