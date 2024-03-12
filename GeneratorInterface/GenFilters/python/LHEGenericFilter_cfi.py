import FWCore.ParameterSet.Config as cms

lheGenericFilter = cms.EDFilter("LHEGenericFilter",
    src = cms.InputTag("source"),
    NumRequired = cms.int32(2),
    ParticleID = cms.vint32(5),
    AcceptLogic = cms.string("LT") # LT meaning < NumRequired, GT >, EQ =, NE !=
)                                
# foo bar baz
# 1pKR1I2fTA86c
