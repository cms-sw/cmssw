import FWCore.ParameterSet.Config as cms

analyzeZToMuMu = cms.EDAnalyzer("PatZToMuMuAnalyzer",
  muons = cms.InputTag("cleanPatMuons"),
  shift = cms.double(1.0)
)
