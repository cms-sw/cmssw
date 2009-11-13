import FWCore.ParameterSet.Config as cms

AlignmentMonitorMuonSystemMap1D = cms.untracked.PSet(
    minTrackPt = cms.double(20.),
    maxTrackPt = cms.double(100.),
    minTrackerHits = cms.int32(10),
    maxTrackerRedChi2 = cms.double(10.),
    allowTIDTEC = cms.bool(True),
    minDT13Hits = cms.int32(8),
    minDT2Hits = cms.int32(4),
    minCSCHits = cms.int32(6),
    writeTemporaryFile = cms.string(""),
    readTemporaryFiles = cms.vstring(),
    doFits = cms.bool(True),
    minFitHits = cms.int32(5),
    )
