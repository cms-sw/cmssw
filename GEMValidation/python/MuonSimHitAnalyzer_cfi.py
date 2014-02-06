import FWCore.ParameterSet.Config as cms

from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching

MuonSimHitAnalyzer = cms.EDAnalyzer("MuonSimHitAnalyzer",
    verbose = cms.untracked.int32(0),
    simInputLabel = cms.untracked.string("g4SimHits"),
    minPt = cms.untracked.double(4.5),
    ntupleTrackChamberDelta = cms.untracked.bool(True),
    ntupleTrackEff = cms.untracked.bool(True),
    simTrackMatching = SimTrackMatching
)
