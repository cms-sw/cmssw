import FWCore.ParameterSet.Config as cms

hltDiEG3023IsoCaloIdClusterShapeSigmawwUnseededFilter = cms.EDFilter("HLTEgammaGenericFilter",
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag("hltDiEG3023IsoCaloIdClusterShapeSigmavvUnseededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesUnseeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(2),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(0),
    thrOverE2EE = cms.vdouble(0),
    thrOverEEB = cms.vdouble(0.04),
    thrOverEEE = cms.vdouble(0.04),
    thrRegularEB = cms.vdouble(64),
    thrRegularEE = cms.vdouble(64),
    useEt = cms.bool(True),
    varTag = cms.InputTag("hltEgammaHGCALIDVarsUnseeded","sigma2ww")
)
