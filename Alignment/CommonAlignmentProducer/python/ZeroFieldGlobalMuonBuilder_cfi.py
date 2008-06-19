import FWCore.ParameterSet.Config as cms

ZeroFieldGlobalMuonBuilder = cms.EDFilter("ZeroFieldGlobalMuonBuilder",
    minTrackerHits = cms.int32(0),
    minPdot = cms.double(0.99),
    minMuonHits = cms.int32(0),
    minDdotP = cms.double(0.99),
    inputMuon = cms.InputTag("cosmicMuons"),
    inputTracker = cms.InputTag("cosmictrackfinderP5"),
    debuggingHistograms = cms.untracked.bool(False)
)


