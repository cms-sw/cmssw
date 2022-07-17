import FWCore.ParameterSet.Config as cms

hltDiEle2312IsoBestGsfNLayerITL1SeededFilter = cms.EDFilter("HLTEgammaGenericFilter",
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag("hltDiEle2312IsoGsfDphiL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(False),
    ncandcut = cms.int32(2),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(0),
    thrOverE2EE = cms.vdouble(0),
    thrOverEEB = cms.vdouble(0),
    thrOverEEE = cms.vdouble(0),
    thrRegularEB = cms.vdouble(3),
    thrRegularEE = cms.vdouble(3),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaBestGsfTrackVarsL1Seeded","NLayerIT")
)
