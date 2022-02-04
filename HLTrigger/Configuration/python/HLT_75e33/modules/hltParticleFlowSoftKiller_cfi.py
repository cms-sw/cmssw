import FWCore.ParameterSet.Config as cms

hltParticleFlowSoftKiller = cms.EDProducer("SoftKillerProducer",
    PFCandidates = cms.InputTag("particleFlowTmp"),
    Rho_EtaMax = cms.double(5.0),
    rParam = cms.double(0.4)
)
