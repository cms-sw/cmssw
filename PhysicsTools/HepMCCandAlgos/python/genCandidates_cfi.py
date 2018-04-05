import FWCore.ParameterSet.Config as cms

genCandidates = cms.EDProducer("HepMCCandidateProducer",
    src = cms.string('generatorSmeared'),
    verbose = cms.untracked.bool(False),
    stableOnly = cms.bool(True),
    excludeList = cms.vstring('nu_e', 
        'nu_mu', 
        'nu_tau', 
        'gamma', 
        'pi0', 
        'K_L0', 
        'n0')
)


