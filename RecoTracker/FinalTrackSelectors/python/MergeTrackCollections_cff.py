import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
# Track filtering and quality.
#   input:    firstStepTracksWithQuality,secStep,thStep
#   output:   generalTracks
#   sequence: trackCollectionMerging

#
mergeFirstTwoSteps = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
mergeFirstTwoSteps.TrackProducer1 = 'firstStepTracksWithQuality'
mergeFirstTwoSteps.TrackProducer2 = 'secStep'
mergeFirstTwoSteps.promoteTrackQuality = True

#
merge2nd3rdTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
merge2nd3rdTracks.TrackProducer1 = 'mergeFirstTwoSteps'
merge2nd3rdTracks.TrackProducer2 = 'thStep'
merge2nd3rdTracks.promoteTrackQuality = True


#
generalTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
generalTracks.TrackProducer1 = 'merge2nd3rdTracks'
generalTracks.TrackProducer2 = 'fourthWithMaterialTracks'
generalTracks.promoteTrackQuality = True




trackCollectionMerging = cms.Sequence(mergeFirstTwoSteps*
                                      merge2nd3rdTracks*
                                      generalTracks)
