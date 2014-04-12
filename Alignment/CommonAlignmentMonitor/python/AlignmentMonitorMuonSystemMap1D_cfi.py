import FWCore.ParameterSet.Config as cms

AlignmentMonitorMuonSystemMap1D = cms.untracked.PSet(
    muonCollectionTag = cms.InputTag(""),
    beamSpotTag = cms.untracked.InputTag("offlineBeamSpot"),
    minTrackPt = cms.double(100.),
    maxTrackPt = cms.double(200.),
    minTrackP = cms.double(0.),
    maxTrackP = cms.double(99999.),
    maxDxy = cms.double(100.),
    minTrackerHits = cms.int32(15),
    maxTrackerRedChi2 = cms.double(10.),
    allowTIDTEC = cms.bool(True),
    minNCrossedChambers = cms.int32(3),
    minDT13Hits = cms.int32(8),
    minDT2Hits = cms.int32(4),
    minCSCHits = cms.int32(6),
    doDT = cms.bool(True),
    doCSC = cms.bool(True),
    useStubPosition = cms.bool(False),
    createNtuple = cms.bool(False)
)
