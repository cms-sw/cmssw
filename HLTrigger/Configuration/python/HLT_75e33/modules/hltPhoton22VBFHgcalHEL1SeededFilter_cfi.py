import FWCore.ParameterSet.Config as cms

# Endcap (HGCAL) H/E < 0.125 E, the loosest Phase-2 value in use; the Run-3 endcap cut is on
# a different variable.
hltPhoton22VBFHgcalHEL1SeededFilter = cms.EDFilter("HLTEgammaGenericQuadraticEtaFilter",
    absEtaLowEdges = cms.vdouble(0.0, 1.0, 1.479, 2.1),
    candTag = cms.InputTag("hltEG22EtL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0, 0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    etaBoundaryEB12 = cms.double(1.0),
    etaBoundaryEE12 = cms.double(2.1),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag("hltFixedGridRhoFastjetAllCaloForEGamma"),
    saveTags = cms.bool(True),
    thrOverE2EB1 = cms.vdouble(0.0),
    thrOverE2EB2 = cms.vdouble(0.0),
    thrOverE2EE1 = cms.vdouble(0.0),
    thrOverE2EE2 = cms.vdouble(0.0),
    thrOverEEB1 = cms.vdouble(0.0),
    thrOverEEB2 = cms.vdouble(0.0),
    thrOverEEE1 = cms.vdouble(0.125),
    thrOverEEE2 = cms.vdouble(0.125),
    thrRegularEB1 = cms.vdouble(9999.0),
    thrRegularEB2 = cms.vdouble(9999.0),
    thrRegularEE1 = cms.vdouble(0.0),
    thrRegularEE2 = cms.vdouble(0.0),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaHGCALIDVarsL1Seeded","hForHOverE")
)
