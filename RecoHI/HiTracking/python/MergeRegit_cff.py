import RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi
hiGeneralAndRegitTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'hiGeneralTracks',
    TrackProducer2 = 'hiRegitTracks',
    promoteTrackQuality = True,
    copyExtras=True
    )
