import FWCore.ParameterSet.Config as cms

# Two Phase-2 L3 muons with pT > 3 GeV matched (dR < 0.3) to the GMTTkMuons of the L1 seeds.
# It plays the role of the L1/L2 matching of the Run-3 HLTMuonDimuonL3Filter, which cannot match
# to P2GT candidates, and it is the PreviousCandTag of the dimuon filters.
hltL3fL1DoubleTkMuonLowMassPreFiltered3 = cms.EDFilter("HLTMuonTrkL1TkMuFilter",
    inputCandCollection = cms.InputTag("hltPhase2L3MuonCandidates"),
    inputMuonCollection = cms.InputTag("hltPhase2L3Muons"),
    maxAbsEta = cms.double(1e+99),
    maxNormalizedChi2 = cms.double(1e+99),
    minMuonHits = cms.int32(-1),
    minMuonStations = cms.int32(1),
    minN = cms.uint32(2),
    minPt = cms.double(3.0),
    minTrkHits = cms.int32(-1),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    l1GTAlgoNames = cms.vstring("pDoubleTkMuon_OS_Er1p5_Dr1p4", "pDoubleTkMuon_4_4_OS_Dr1p2"),
    saveTags = cms.bool(True)
)
