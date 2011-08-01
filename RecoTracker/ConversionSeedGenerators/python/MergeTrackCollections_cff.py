import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi
# Track filtering and quality.
#   input:    zeroStepTracksWithQuality,preMergingFirstStepTracksWithQuality,secStep,thStep,pixellessStep
#   output:   generalTracks
#   sequence: trackCollectionMerging

#

firstStepTracksWithQuality = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'zeroStepTracksWithQuality',
    TrackProducer2 = 'preMergingFirstStepTracksWithQuality',
    promoteTrackQuality = False
    )


merge2nd3rdTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'secStep',
    TrackProducer2 = 'thStep',
    promoteTrackQuality = True
    )

merge4th5thTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'pixellessStep',
    TrackProducer2 = 'tobtecStep',
    promoteTrackQuality = True
    )

iterTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'merge2nd3rdTracks',
    TrackProducer2 = 'merge4th5thTracks',
    promoteTrackQuality = True
    )

mergeConversionTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'sixthStep',
    TrackProducer2 = 'seventhStep',
    promoteTrackQuality = True
    )

newIterTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'iterTracks',
    TrackProducer2 = 'mergeConversionTracks',
    promoteTrackQuality = True
    )

generalTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'firstStepTracksWithQuality',
    TrackProducer2 = 'newIterTracks',
    promoteTrackQuality = True,
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(True)
    )


trackCollectionMerging = cms.Sequence(merge2nd3rdTracks*
                                      merge4th5thTracks*
                                      iterTracks*
                                      mergeConversionTracks*
                                      newIterTracks*
                                      generalTracks)
