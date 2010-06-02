import FWCore.ParameterSet.Config as cms

hltMuonValidator = cms.EDAnalyzer("HLTMuonValidator",

    hltProcessName = cms.string("HLT"),
    hltPathsToCheck = cms.vstring("HLT_[^H_]*Mu[^_]*$"),

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

    # set criteria for matching
    cutsDr      = cms.vdouble(0.4, 0.4, 0.015),

    # set cuts on generated and reconstructed muons
    genMuonCut  = cms.string("abs(pdgId) == 13 && status == 1"),
    recMuonCut  = cms.string("isGlobalMuon"),

)
