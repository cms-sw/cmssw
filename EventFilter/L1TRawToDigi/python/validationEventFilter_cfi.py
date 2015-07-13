
import FWCore.ParameterSet.Config as cms

validationEventFilter = cms.EDFilter(
    "L1TValidationEventFilter",
    period = cms.untracked.int32(107)
)
