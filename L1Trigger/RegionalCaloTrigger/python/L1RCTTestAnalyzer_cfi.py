import FWCore.ParameterSet.Config as cms

L1RCTTestAnalyzer = cms.EDAnalyzer("L1RCTTestAnalyzer",
    showEmCands = cms.untracked.bool(True),
    showRegionSums = cms.untracked.bool(True)
)


