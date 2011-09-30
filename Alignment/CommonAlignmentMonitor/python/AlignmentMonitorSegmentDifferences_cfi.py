import FWCore.ParameterSet.Config as cms

AlignmentMonitorSegmentDifferences = cms.untracked.PSet(
    minTrackPt = cms.double(50.),
    minTrackerHits = cms.int32(15),
    maxTrackerRedChi2 = cms.double(10.),
    allowTIDTEC = cms.bool(True),
    minDT13Hits = cms.int32(8),
    minDT2Hits = cms.int32(4),
    minCSCHits = cms.int32(6)
    )
