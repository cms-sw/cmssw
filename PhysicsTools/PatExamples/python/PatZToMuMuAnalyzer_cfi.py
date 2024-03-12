import FWCore.ParameterSet.Config as cms

analyzeZToMuMu = cms.EDAnalyzer("PatZToMuMuAnalyzer",
  muons = cms.InputTag("cleanPatMuons"),
  shift = cms.double(1.0)
)
# foo bar baz
# A3dzu1aMziAxl
# KhTXBanNOiOBh
