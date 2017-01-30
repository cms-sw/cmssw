
import FWCore.ParameterSet.Config as cms

validationEventFilter = cms.EDFilter(
    "L1TValidationEventFilter",
    src       = cms.InputTag("tcdsDigis","triggerCount"),
    period    = cms.int32(107)
)
