import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
# Track filtering and quality.
#   input:    zeroStepTracksWithQuality,preMergingFirstStepTracksWithQuality,secStep,thStep,pixellessStep
#   output:   generalTracks
#   sequence: trackCollectionMerging

#

firstStepTracksWithQuality = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone(
    TrackProducer1 = 'zeroStepTracksWithQuality',
    TrackProducer2 = 'preMergingFirstStepTracksWithQuality',
    promoteTrackQuality = False
    )


merge2nd3rdTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone(
    TrackProducer1 = 'secStep',
    TrackProducer2 = 'thStep',
    promoteTrackQuality = True
    )

merge4th5thTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone(
    TrackProducer1 = 'pixellessStep',
    TrackProducer2 = 'tobtecStep',
    promoteTrackQuality = True
    )

iterTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone(
    TrackProducer1 = 'merge2nd3rdTracks',
    TrackProducer2 = 'merge4th5thTracks',
    promoteTrackQuality = True
    )

generalTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone(
    TrackProducer1 = 'firstStepTracksWithQuality',
    TrackProducer2 = 'iterTracks',
    promoteTrackQuality = True
    )

trackCollectionMerging = cms.Sequence(merge2nd3rdTracks*
                                      merge4th5thTracks*
                                      iterTracks*
                                      generalTracks)
