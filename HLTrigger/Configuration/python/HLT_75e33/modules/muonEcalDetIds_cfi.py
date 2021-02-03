import FWCore.ParameterSet.Config as cms

muonEcalDetIds = cms.EDProducer("InterestingEcalDetIdProducer",
    inputCollection = cms.InputTag("muons1stStep")
)
