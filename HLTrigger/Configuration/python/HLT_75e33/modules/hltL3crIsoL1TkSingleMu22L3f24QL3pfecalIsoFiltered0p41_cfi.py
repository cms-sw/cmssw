import FWCore.ParameterSet.Config as cms

hltL3crIsoL1TkSingleMu22L3f24QL3pfecalIsoFiltered0p41 = cms.EDFilter("HLTMuonGenericFilter",
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag("hltL3fL1TkSingleMu22L3Filtered24Q"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltPhase2L3MuonCandidates"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(-1.0),
    thrOverE2EE = cms.vdouble(-1.0),
    thrOverEEB = cms.vdouble(0.41),
    thrOverEEE = cms.vdouble(0.41),
    thrRegularEB = cms.vdouble(-1.0),
    thrRegularEE = cms.vdouble(-1.0),
    useEt = cms.bool(True),
    varTag = cms.InputTag("hltPhase2L3MuonsEcalIsodR0p3dRVeto0p000")
)
