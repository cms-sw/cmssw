import FWCore.ParameterSet.Config as cms

particleFlowPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
    src = cms.InputTag("particleFlowTmp")
)
# foo bar baz
