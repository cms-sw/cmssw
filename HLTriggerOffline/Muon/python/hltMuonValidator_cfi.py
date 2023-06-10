import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

hltMuonValidator = DQMEDAnalyzer('HLTMuonValidator',

    hltProcessName = cms.string("HLT"),
    hltPathsToCheck = cms.vstring(
        "HLT_(HighPt)?(L[12])?(Iso)?(Tk)?Mu[0-9]*(Open)?(_NoVertex)?(_eta2p1)?(_v[0-9]*)?$",
        "HLT_Mu17_NoFilters?(_v[0-9]*)?$",
        "HLT_Dimuon0_Jpsi(_v[0-9]*)?$",
        "HLT_Dimuon13_Jpsi_Barrel(_v[0-9]*)?$",
        ),

    genParticleLabel = cms.string("genParticles"       ),
        recMuonLabel = cms.string("muons"              ),

    parametersTurnOn = cms.vdouble(0,
                                   1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                   22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                                   45, 50, 55, 60, 65, 70,
                                   80, 100, 200, 500, 1000, 2000,
                                   ), 
    parametersEta      = cms.vdouble(48, -2.400, 2.400),
    parametersPhi      = cms.vdouble(50, -3.142, 3.142),

    # set criteria for matching at L1, L2, L3
    cutsDr = cms.vdouble(0.4, 0.4, 0.015),
    # parameters for attempting an L1 match using a propagator
    maxDeltaPhi = cms.double(0.4),
    maxDeltaR   = cms.double(0.4),
    useSimpleGeometry = cms.bool(True),
    useTrack = cms.string("none"),
    useState = cms.string("atVertex"),
    useStation2 = cms.bool(True),
    fallbackToME1 = cms.bool(False),
    cosmicPropagationHypothesis = cms.bool(False),
    useMB2InOverlap = cms.bool(False),
    propagatorAlong = cms.ESInputTag("", "hltESPSteppingHelixPropagatorAlong"),
    propagatorAny = cms.ESInputTag("", "SteppingHelixPropagatorAny"),
    propagatorOpposite = cms.ESInputTag("", "hltESPSteppingHelixPropagatorOpposite"),
    # set cuts on generated and reconstructed muons
    genMuonCut  = cms.string("abs(pdgId) == 13 && status == 1"),
    recMuonCut  = cms.string("isGlobalMuon"),
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(hltMuonValidator,
                       hltPathsToCheck = cms.vstring(
                           "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_FromL1TkMuon(_v[0-9]*)?$",
                           "HLT_Mu37_Mu27_FromL1TkMuon(_v[0-9]*)?$",
                           "HLT_Mu50_FromL1TkMuon(_v[0-9]*)?$"
                       )
)
