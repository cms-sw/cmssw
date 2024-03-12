
import FWCore.ParameterSet.Config as cms

validationEventFilter = cms.EDFilter(
    "L1TValidationEventFilter",
    tcsdRecord = cms.InputTag("tcdsDigis","tcsdRecord"),
    period     = cms.int32(107)
)
# foo bar baz
# Lnayml2KI3fHj
# LQK3C43UOo9Mx
