import FWCore.ParameterSet.Config as cms

hltBoolFalse = cms.EDFilter("HLTBool",
    result = cms.bool(False)
)
