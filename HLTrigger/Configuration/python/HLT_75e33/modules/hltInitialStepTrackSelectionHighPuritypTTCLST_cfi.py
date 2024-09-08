import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTrackSelectionHighPurity_cfi import hltInitialStepTrackSelectionHighPurity as _hltInitialStepTrackSelectionHighPurity
hltInitialStepTrackSelectionHighPuritypTTCLST = _hltInitialStepTrackSelectionHighPurity.clone(
    originalMVAVals = "hltInitialStepTrackCutClassifierpTTCLST:MVAValues",
    originalQualVals = "hltInitialStepTrackCutClassifierpTTCLST:QualityMasks",
    originalSource = "hltInitialStepTrackspTTCLST"
)

