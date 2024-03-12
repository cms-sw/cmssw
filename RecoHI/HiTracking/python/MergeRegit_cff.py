from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi
hiGeneralAndRegitTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'hiGeneralTracks',
    TrackProducer2 = 'hiRegitTracks',
    promoteTrackQuality = True,
    copyExtras=True
    )
# foo bar baz
# olriZ7Rb3N8t3
