import FWCore.ParameterSet.Config as cms

# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_TkMu.
# No L1 matching here (empty l1GTAlgoNames), as the default L1 menu is seeded by a
# jet-only seed. customizeHLTforL1P2GTv0p0_2026 sets l1GTAlgoNames to pDoubleJet_TkMu.
hltL3fL1TkMuVBFFiltered5 = cms.EDFilter("HLTMuonTrkL1TkMuFilter",
    inputCandCollection = cms.InputTag("hltPhase2L3MuonCandidates"),
    inputMuonCollection = cms.InputTag("hltPhase2L3Muons"),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    l1GTAlgoNames = cms.vstring(),
    maxAbsEta = cms.double(2.4),
    maxNormalizedChi2 = cms.double(1e+99),
    minMuonHits = cms.int32(-1),
    minMuonStations = cms.int32(1),
    minN = cms.uint32(1),
    minPt = cms.double(5.0),
    minTrkHits = cms.int32(-1),
    saveTags = cms.bool(True)
)
