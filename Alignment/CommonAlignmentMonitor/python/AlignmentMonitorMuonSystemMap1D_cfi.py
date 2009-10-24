import FWCore.ParameterSet.Config as cms

AlignmentMonitorMuonSystemMap1D = cms.untracked.PSet(
    minTrackPt = cms.double(100.),
    maxTrackPt = cms.double(200.),
    minTrackerHits = cms.int32(15),
    maxTrackerRedChi2 = cms.double(10.),
    allowTIDTEC = cms.bool(True),
    minDT13Hits = cms.int32(8),
    minDT2Hits = cms.int32(4),
    minCSCHits = cms.int32(6)
    )
