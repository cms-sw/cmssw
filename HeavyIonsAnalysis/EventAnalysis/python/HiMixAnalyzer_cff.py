import FWCore.ParameterSet.Config as cms

mixAnalyzer = cms.EDAnalyzer('HiMixValidation',
                             doRECO = cms.untracked.bool(False),
                             jetSrc = cms.untracked.InputTag('akPu4CaloJets')
)



