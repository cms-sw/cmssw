import FWCore.ParameterSet.Config as cms

# a dummy track collection
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
jetCoreRegionalStepTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (),
    hasSelector=cms.vint32(),
    selectedTrackQuals = cms.VInputTag(),
    copyExtras = True
    )

# a dummy track selector
import RecoTracker.IterativeTracking.JetCoreRegionalStep_cff
jetCoreRegionalStepSelector = RecoTracker.IterativeTracking.JetCoreRegionalStep_cff.jetCoreRegionalStepSelector.clone()
jetCoreRegionalStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

# a dummy sequence
JetCoreRegionalStep = cms.Sequence(jetCoreRegionalStepTracks*
                                   jetCoreRegionalStepSelector)
