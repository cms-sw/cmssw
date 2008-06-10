import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
# Track filtering and quality.
#   input:    firstStepTracksWithQuality,secStep,thStep
#   output:   generalTracks
#   sequence: trackCollectionMerging
mergeFirstTwoSteps = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
generalTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
trackCollectionMerging = cms.Sequence(mergeFirstTwoSteps*generalTracks)
#
mergeFirstTwoSteps.TrackProducer1 = 'firstStepTracksWithQuality'
mergeFirstTwoSteps.TrackProducer2 = 'secStep'
mergeFirstTwoSteps.promoteTrackQuality = True
#
generalTracks.TrackProducer1 = 'mergeFirstTwoSteps'
generalTracks.TrackProducer2 = 'thStep'
generalTracks.promoteTrackQuality = True
