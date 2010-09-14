import FWCore.ParameterSet.Config as cms

analyzePatTopselection = cms.EDAnalyzer("PatTopSelectionAnalyzer",
    elecs = cms.untracked.InputTag("selectedPatElectrons"),
    muons = cms.untracked.InputTag("selectedPatMuons"),                                             
    jets  = cms.untracked.InputTag("selectedPatJets"),
    met   = cms.untracked.InputTag("patMETs")
)
