import FWCore.ParameterSet.Config as cms

# One Phase-2 L3 muon with pT > 9 GeV matched (dR < 0.3) to the GMTTkMuon of pSingleTkMu_Barrel.
# It plays the role of the Run-3 L1/L2 matching, which cannot match to P2GT candidates, and it is
# the PreviousCandTag of hltL3fL1TkSingleMuBarrelIP6L3Filtered9.
hltL3fL1TkSingleMuBarrelL3Filtered9 = cms.EDFilter("HLTMuonTrkL1TkMuFilter",
    inputCandCollection = cms.InputTag("hltPhase2L3MuonCandidates"),
    inputMuonCollection = cms.InputTag("hltPhase2L3Muons"),
    maxAbsEta = cms.double(1e+99),
    maxNormalizedChi2 = cms.double(1e+99),
    minMuonHits = cms.int32(-1),
    minMuonStations = cms.int32(1),
    minN = cms.uint32(1),
    minPt = cms.double(9.0),
    minTrkHits = cms.int32(-1),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    l1GTAlgoNames = cms.vstring("pSingleTkMu_Barrel"),
    saveTags = cms.bool(True)
)
