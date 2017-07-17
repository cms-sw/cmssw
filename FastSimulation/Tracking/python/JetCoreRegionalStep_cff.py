import FWCore.ParameterSet.Config as cms

# a dummy track collection
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
jetCoreRegionalStepTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (),
    hasSelector=cms.vint32(),
    selectedTrackQuals = cms.VInputTag(),
    copyExtras = True
    )

# a dummy track selector
import RecoTracker.IterativeTracking.JetCoreRegionalStep_cff
jetCoreRegionalStep = RecoTracker.IterativeTracking.JetCoreRegionalStep_cff.jetCoreRegionalStep.clone()
jetCoreRegionalStep.vertices = "firstStepPrimaryVerticesBeforeMixing"

# a dummy sequence
JetCoreRegionalStep = cms.Sequence(jetCoreRegionalStepTracks*
                                   jetCoreRegionalStep)
