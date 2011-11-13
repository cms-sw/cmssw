import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi
# Track filtering and quality.
#   input:    zeroStepTracksWithQuality,preMergingFirstStepTracksWithQuality,secStep,thStep,pixellessStep
#   output:   generalTracks
#   sequence: trackCollectionMerging

#

thirdAndFourthSteps = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'hiMixedTripletSelectedTracks',
    TrackProducer2 = 'hiPixelPairSelectedTracks',
    promoteTrackQuality = True
    )


iterTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'hiSecondPixelTripletSelectedTracks',
    TrackProducer2 = 'thirdAndFourthSteps',
    promoteTrackQuality = True
    )


hiGeneralTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'hiSelectedTracks',
    TrackProducer2 = 'iterTracks',
    promoteTrackQuality = True,
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(True)
    )


trackCollectionMerging = cms.Sequence(
    thirdAndFourthSteps*
    iterTracks*
    hiGeneralTracks
    )
