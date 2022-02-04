import FWCore.ParameterSet.Config as cms

hltL3fL1TkSingleMu22L3Filtered24Q = cms.EDFilter("HLTMuonTrkL1TkMuFilter",
    inputCandCollection = cms.InputTag("hltPhase2L3MuonCandidates"),
    inputMuonCollection = cms.InputTag("hltPhase2L3Muons"),
    maxAbsEta = cms.double(1e+99),
    maxNormalizedChi2 = cms.double(1e+99),
    minMuonHits = cms.int32(-1),
    minMuonStations = cms.int32(1),
    minN = cms.uint32(1),
    minPt = cms.double(24.0),
    minTrkHits = cms.int32(-1),
    previousCandTag = cms.InputTag("hltL1TkSingleMuFiltered22"),
    saveTags = cms.bool(True)
)
