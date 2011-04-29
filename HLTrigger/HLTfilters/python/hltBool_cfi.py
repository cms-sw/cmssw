import FWCore.ParameterSet.Config as cms

hltBool = cms.EDFilter("HLTBool",
    result = cms.bool(False)
)


