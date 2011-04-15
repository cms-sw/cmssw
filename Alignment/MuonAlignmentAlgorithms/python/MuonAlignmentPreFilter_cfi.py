import FWCore.ParameterSet.Config as cms

MuonAlignmentPreFilter = cms.EDFilter("MuonAlignmentPreFilter",
  tracksTag = cms.InputTag("ALCARECOMuAlCalIsolatedMu:GlobalMuon"),
  minTrackPt = cms.double(20.),
  minTrackP = cms.double(0.),
  minTrackerHits = cms.int32(10),
  minDTHits = cms.int32(6),
  minCSCHits = cms.int32(4),
  allowTIDTEC = cms.bool(True),
  minTrackEta = cms.double(-2.4),
  maxTrackEta = cms.double(2.4)
)

