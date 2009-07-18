import FWCore.ParameterSet.Config as cms

qcdHighPtDQM = cms.EDAnalyzer("QcdHighPtDQM",
                                     jetTag = cms.untracked.InputTag("sisCone5CaloJets"),
                                     metTag = cms.untracked.InputTag("met")
)

