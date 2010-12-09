import FWCore.ParameterSet.Config as cms

hltMuonValidator = cms.EDAnalyzer("HLTMuonValidator",

    hltProcessName = cms.string("HLT"),
    hltPathsToCheck = cms.vstring(
        "HLT_(L[12])?(Double)?(Iso)?Mu[0-9]*(Open)?(_NoVertex)?(_v[0-9]*)?$",
        "NoFilters",   # special value for looking at objects without filters
        ),

    genParticleLabel = cms.string("genParticles"       ),
        recMuonLabel = cms.string("muons"              ),
         l1CandLabel = cms.string("hltL1extraParticles"),
         l2CandLabel = cms.string("hltL2MuonCandidates"),
         l3CandLabel = cms.string("hltL3MuonCandidates"),

    parametersTurnOn = cms.vdouble(0,
                                   1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                   22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                                   45, 50, 55, 60, 65, 70,
                                   80, 100, 200, 500
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

    # set cuts on generated and reconstructed muons
    genMuonCut  = cms.string("abs(pdgId) == 13 && status == 1"),
    recMuonCut  = cms.string("isGlobalMuon"),

)
