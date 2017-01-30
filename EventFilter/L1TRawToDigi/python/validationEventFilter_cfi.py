
import FWCore.ParameterSet.Config as cms

validationEventFilter = cms.EDFilter(
    "L1TValidationEventFilter",
    src       = cms.InputTag("tcdsDigis"),
    period    = cms.untracked.int32(107)
)
