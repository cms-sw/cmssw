import FWCore.ParameterSet.Config as cms

mixAnalyzer = cms.EDAnalyzer('HiMixValidation',
                             doRECO = cms.untracked.bool(False),
                             genpSrc = cms.untracked.InputTag('genParticles'),
                             jetSrc = cms.untracked.InputTag('akPu4CaloJets')
)



