import FWCore.ParameterSet.Config as cms

hltParticleFlowCHS = cms.EDProducer("FwdPtrRecoPFCandidateConverter",
    src = cms.InputTag("pfNoPileUpJME")
)
