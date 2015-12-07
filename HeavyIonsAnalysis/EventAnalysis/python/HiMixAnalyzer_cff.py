import FWCore.ParameterSet.Config as cms

process.mixAnalyzer = cms.EDAnalyzer('HiMixValidation',
                                     jetSrc = cms.untracked.InputTag('akPu4CaloJets')
)



