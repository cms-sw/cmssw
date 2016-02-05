
import FWCore.ParameterSet.Config as cms

validationEventFilter = cms.EDFilter(
    "L1TValidationEventFilter",
    inputTag  = cms.InputTag("rawDataCollector"),
    period    = cms.untracked.int32(107)
)
