import FWCore.ParameterSet.Config as cms

isolatedParticlesGeneratedJets = cms.EDAnalyzer("IsolatedParticlesGeneratedJets",
                                                Debug            = cms.untracked.bool(True),
                                                JetSource        = cms.InputTag("ak5GenJets"),
                                                ParticleSource   = cms.InputTag("genParticles")
)
