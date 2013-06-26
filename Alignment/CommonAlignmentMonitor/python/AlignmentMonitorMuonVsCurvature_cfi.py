import FWCore.ParameterSet.Config as cms

AlignmentMonitorMuonVsCurvature = cms.untracked.PSet(
    muonCollectionTag = cms.InputTag(""),
    beamSpotTag = cms.untracked.InputTag("offlineBeamSpot"),
    minTrackPt = cms.double(20.),
    minTrackP = cms.double(0.),
    minTrackerHits = cms.int32(10),
    maxTrackerRedChi2 = cms.double(100.),
    allowTIDTEC = cms.bool(True),
    minNCrossedChambers = cms.int32(3),
    maxDxy = cms.double(10.),
    minDT13Hits = cms.int32(8),
    minDT2Hits = cms.int32(4),
    minCSCHits = cms.int32(6),
    layer = cms.int32(2),
    propagator = cms.string("SmartPropagatorAnyRK"),
    doDT = cms.bool(True),
    doCSC = cms.bool(True)
)
