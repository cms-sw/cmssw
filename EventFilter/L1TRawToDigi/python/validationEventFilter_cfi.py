
import FWCore.ParameterSet.Config as cms

validationEventFilter = cms.EDFilter(
    "L1TValidationEventFilter",
    tcsdRecord = cms.InputTag("tcdsDigis","tcsdRecord"),
    period     = cms.int32(107)
)
