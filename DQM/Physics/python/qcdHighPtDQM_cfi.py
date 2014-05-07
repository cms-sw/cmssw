import FWCore.ParameterSet.Config as cms

qcdHighPtDQM = cms.EDAnalyzer("QcdHighPtDQM",
                                     jetTag = cms.untracked.InputTag("ak4CaloJets"),
                                      metTag1 = cms.untracked.InputTag("met"),
                                      metTag2 = cms.untracked.InputTag("metHO"),
                                      metTag3 = cms.untracked.InputTag("metNoHF"),
                                      metTag4 = cms.untracked.InputTag("metNoHFHO")
)

