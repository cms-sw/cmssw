import FWCore.ParameterSet.Config as cms

hltBoolEnd = cms.EDFilter("HLTBool",
    result = cms.bool(True)
)
