import FWCore.ParameterSet.Config as cms

# Two Phase-2 L3 muons matched (dR < 0.3) to the GMTTkMuons of the L1 seeds, no pT cut as in the
# Run-3 path. It plays the role of the L1/L2 matching of the Run-3 HLTMuonDimuonL3Filter, which
# cannot match to P2GT candidates, and it is the PreviousCandTag of hltDimuon0UpsilonL3Filtered.
hltL3fL1DoubleTkMuonUpsilonPreFiltered0 = cms.EDFilter("HLTMuonTrkL1TkMuFilter",
    inputCandCollection = cms.InputTag("hltPhase2L3MuonCandidates"),
    inputMuonCollection = cms.InputTag("hltPhase2L3Muons"),
    maxAbsEta = cms.double(1e+99),
    maxNormalizedChi2 = cms.double(1e+99),
    minMuonHits = cms.int32(-1),
    minMuonStations = cms.int32(1),
    minN = cms.uint32(2),
    minPt = cms.double(0.0),
    minTrkHits = cms.int32(-1),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    l1GTAlgoNames = cms.vstring("pDoubleTkMuon_OS_Er1p5_Dr1p4", "pDoubleTkMuon_4p5_4p5_OS_Er2_Mass7to18"),
    saveTags = cms.bool(True)
)
