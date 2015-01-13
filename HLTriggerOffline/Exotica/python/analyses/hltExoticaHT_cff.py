import FWCore.ParameterSet.Config as cms

HTPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v",
        "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v",
        "HLT_PFHT750_4Jet_v", # Run2
        "HLT_PFHT650_4Jet_v", # Run2
        "HLT_PFHT550_4Jet_v", # Run2
        "HLT_PFHT900_v",      # Run2
        "HLT_PFHT650_v",
        "HLT_HT900_v",        # Run2
        "HLT_HT300_v",        # Run2
        "HLT_ECALHT800_v"     # Run2 7e33
        #"HLT_HT750_v"        # Run1 (frozenHLT)
        ),
    recPFMHTLabel  = cms.InputTag("recoExoticaValidationHT"),
    recPFJetLabel  = cms.InputTag("ak4PFJets"),
    # -- Analysis specific cuts
    MET_genCut      = cms.string("sumEt > 75"),
    MET_recCut      = cms.string("sumEt > 75"),
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(0, 50, 100, 150,
                                   200, 220, 240, 260, 280, 300,
                                   320, 340, 360, 380, 400,
                                   420, 440, 460, 480, 500,
                                   520, 540, 560, 580, 600,
                                   620, 640, 660, 680, 700,
                                   750, 800, 850, 900, 950, 1000,
                                   1100, 1200, 1300, 1400, 1500)
)
