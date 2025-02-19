import FWCore.ParameterSet.Config as cms

#from RecoHI.HiTracking.hiMultiTrackSelector_cfi import *

# Track selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiInitialStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiGlobalPrimTracks',
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiInitialStepLoose',
    keepAllTracks = True  # Make an exception for the 1st iteration
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiInitialStepTight',
    preFilterName = 'hiInitialStepLoose',
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiInitialStep',
    preFilterName = 'hiInitialStepTight',
    ),
    ) #end of vpset
    ) #end of clone  



# using the tracklist merger with one collection simply applies the quality flags
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiSelectedTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('hiGlobalPrimTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiInitialStepSelector","hiInitialStep")),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )

#complete sequence
hiTracksWithQuality = cms.Sequence(hiInitialStepSelector
                                   * hiSelectedTracks)
