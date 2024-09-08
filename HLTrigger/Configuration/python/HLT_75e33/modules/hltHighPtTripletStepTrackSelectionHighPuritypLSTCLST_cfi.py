import FWCore.ParameterSet.Config as cms

from ..modules.hltHighPtTripletStepTrackSelectionHighPurity_cfi import hltHighPtTripletStepTrackSelectionHighPurity as _hltHighPtTripletStepTrackSelectionHighPurity
hltHighPtTripletStepTrackSelectionHighPuritypLSTCLST = _hltHighPtTripletStepTrackSelectionHighPurity.clone(
    originalMVAVals = "hltHighPtTripletStepTrackCutClassifierpLSTCLST:MVAValues",
    originalQualVals = "hltHighPtTripletStepTrackCutClassifierpLSTCLST:QualityMasks",
    originalSource = "hltHighPtTripletStepTrackspLSTCLST"
)

