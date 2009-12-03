import FWCore.ParameterSet.Config as cms

hltMuonValidator = cms.EDAnalyzer("HLTMuonValidator",

    hltProcessName = cms.string("HLT"),
    hltPathsToCheck = cms.vstring("HLT_[^_]*Mu[^_]*$"),

    parametersTurnOn = cms.vdouble(0,
                                   1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                   22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                                   45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100,
                                   110, 120, 140, 170, 200, 250, 300, 380, 500
                       ), 
    parametersEta      = cms.vdouble(48, -2.400, 2.400),
    parametersPhi      = cms.vdouble(50, -3.142, 3.142),

    # Set cuts placed on the generated muons and matching criteria
    # Use pt cut just below 10 to allow through SingleMuPt10 muons  
    cutMinPt           = cms.double(9.9),
    cutMotherId        = cms.uint32(0),
    cutsDr             = cms.vdouble(0.4, 0.4, 0.015),
    TriggerResultLabel = cms.InputTag("TriggerResults","","HLT"),

)
