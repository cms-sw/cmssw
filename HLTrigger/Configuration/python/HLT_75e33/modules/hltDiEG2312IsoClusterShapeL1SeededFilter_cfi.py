import FWCore.ParameterSet.Config as cms

hltDiEG2312IsoClusterShapeL1SeededFilter = cms.EDFilter("HLTEgammaGenericFilter",
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag("hltDiEG12EtL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(2),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(0),
    thrOverE2EE = cms.vdouble(0),
    thrOverEEB = cms.vdouble(0),
    thrOverEEE = cms.vdouble(0),
    thrRegularEB = cms.vdouble(0.017),
    thrRegularEE = cms.vdouble(0.017),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaClusterShapeL1Seeded","sigmaIEtaIEta5x5")
)
