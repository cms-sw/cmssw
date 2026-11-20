import FWCore.ParameterSet.Config as cms

# Very loose relative track isolation (< 0.4), as in the Run-3 Mu3_TrkIsoVVL leg.
hltL3fL1TkMuVBF5RelTrkIsoVVLFiltered0p4 = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltPhase2L3MuonCandidates"),
    DepTag = cms.VInputTag("hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p4"),
    IsolatorPSet = cms.PSet(),
    MinN = cms.int32(1),
    PreviousCandTag = cms.InputTag("hltL3fL1TkMuVBFFiltered5"),
    saveTags = cms.bool(True)
)
