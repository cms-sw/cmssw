import FWCore.ParameterSet.Config as cms

hltEle26WP70ClusterShapeSigmavvL1SeededFilter = cms.EDFilter("HLTEgammaGenericFilter",
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag("hltEle26WP70ClusterShapeL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(0),
    thrOverE2EE = cms.vdouble(0),
    thrOverEEB = cms.vdouble(0.0008),
    thrOverEEE = cms.vdouble(0.0008),
    thrRegularEB = cms.vdouble(0.64),
    thrRegularEE = cms.vdouble(0.64),
    useEt = cms.bool(True),
    varTag = cms.InputTag("hltEgammaHGCALIDVarsL1Seeded","sigma2vv")
)
