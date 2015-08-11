import FWCore.ParameterSet.Config as cms

HTPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v",
        "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v",
        #"HLT_PFHT750_4Jet_v", # Run2
        "HLT_PFHT750_4Jet_Pt50_v",
        "HLT_PFHT650_4Jet_v", # Run2
        "HLT_PFHT550_4Jet_v", # Run2
        #"HLT_PFHT900_v",      # Run2
        "HLT_PFHT800_v",
        "HLT_PFHT650_v",
        "HLT_HT900_v",        # Run2
        "HLT_HT300_v",        # Run2
        "HLT_HT450to470_v",        # HT Parking
        "HLT_HT470to500_v",        # HT Parking
        "HLT_HT500to550_v",        # HT Parking
        "HLT_HT550to650_v",        # HT Parking
        "HLT_HT650_v",        # HT Parking
        "HLT_ECALHT800_v",     # Run2 7e33
        "HLT_Photon90_CaloIdL_PFHT600_v" # 50ns backup menu
        #"HLT_HT750_v"        # Run1 (frozenHLT)
        ),
    recPFMHTLabel  = cms.InputTag("recoExoticaValidationHT"),
    recPFJetLabel  = cms.InputTag("ak4PFJets"),
    # -- Analysis specific cuts
    MET_genCut      = cms.string("sumEt > 75"),
    MET_recCut      = cms.string("sumEt > 75"),
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 
                                   500, 550, 600, 650, 700, 800, 900, 1000
                                   )
)
