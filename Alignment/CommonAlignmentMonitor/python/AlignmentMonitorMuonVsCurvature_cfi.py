import FWCore.ParameterSet.Config as cms

AlignmentMonitorMuonVsCurvature = cms.untracked.PSet(
    minTrackPt = cms.double(20.),
    minTrackerHits = cms.int32(10),
    maxTrackerRedChi2 = cms.double(100.),
    allowTIDTEC = cms.bool(True),
    maxDxy = cms.double(10.),
    minDT13Hits = cms.int32(8),
    minDT2Hits = cms.int32(4),
    minCSCHits = cms.int32(6),
    layer = cms.int32(2),
    propagator = cms.string("SmartPropagatorAnyRK"),
    )
