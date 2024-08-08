import FWCore.ParameterSet.Config as cms

hltParticleFlowPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
    src = cms.InputTag("hltParticleFlowTmp")
)
