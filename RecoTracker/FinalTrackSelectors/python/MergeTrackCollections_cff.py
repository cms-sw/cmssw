import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
# Track filtering and quality.
#   input:    preFilterCmsTracks
#   output:   generalTracks
#   sequence: tracksWithQuality
generalTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
trackCollectionMerging = cms.Sequence(generalTracks)
generalTracks.TrackProducer1 = 'firstStepTracksWithQuality'
generalTracks.TrackProducer2 = 'secStep'
generalTracks.promoteTrackQuality = True

