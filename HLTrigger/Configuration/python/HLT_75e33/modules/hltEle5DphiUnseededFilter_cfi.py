import FWCore.ParameterSet.Config as cms

hltEle5DphiUnseededFilter = cms.EDFilter("HLTEgammaGenericFilter",
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag("hltEgammaCandidatesWrapperUnseeded"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesUnseeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(-1.0),
    thrOverE2EE = cms.vdouble(-1.0),
    thrOverEEB = cms.vdouble(-1.0),
    thrOverEEE = cms.vdouble(-1.0),
    thrRegularEB = cms.vdouble(10),
    thrRegularEE = cms.vdouble(10),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaGsfTrackVarsUnseeded","Dphi")
)
