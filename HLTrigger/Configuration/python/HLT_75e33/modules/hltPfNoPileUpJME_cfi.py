import FWCore.ParameterSet.Config as cms

hltPfNoPileUpJME = cms.EDProducer("TPPFCandidatesOnPFCandidates",
    bottomCollection = cms.InputTag("hltParticleFlowPtrs"),
    enable = cms.bool(True),
    name = cms.untracked.string('pileUpOnPFCandidates'),
    topCollection = cms.InputTag("hltPfPileUpJME")
)
