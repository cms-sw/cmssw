import FWCore.ParameterSet.Config as cms

hltL3crIsoL1TkSingleMu22L3f24QL3trkIsoRegionalNewFiltered0p07EcalHcalHgcalTrk = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltPhase2L3MuonCandidates"),
    DepTag = cms.VInputTag("hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p07"),
    IsolatorPSet = cms.PSet(

    ),
    MinN = cms.int32(1),
    PreviousCandTag = cms.InputTag("hltL3crIsoL1TkSingleMu22L3f24QL3pfhgcalIsoFiltered4p70"),
    saveTags = cms.bool(True)
)
