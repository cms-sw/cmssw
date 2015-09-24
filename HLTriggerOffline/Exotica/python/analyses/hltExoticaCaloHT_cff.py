import FWCore.ParameterSet.Config as cms

CaloHTPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_HT900_v",         # Run2
        "HLT_HT300_v",         # Run2
        "HLT_ECALHT800_v",     # Run2 7e33
        "HLT_Photon90_CaloIdL_PFHT600_v" # 50ns backup menu
        "HLT_HT650_v",           
        "HLT_HT450to470_v",        # HT Parking
        "HLT_HT470to500_v",        # HT Parking
        "HLT_HT500to550_v",        # HT Parking
        "HLT_HT550to650_v",        # HT Parking
        "DST_HT250_CaloScouting_v", # scouting
        "DST_CaloJet40_CaloScouting_v",
        "DST_L1HTT125ORHTT150ORHTT175_CaloScouting_v"
        ),
    recCaloMHTLabel  = cms.InputTag("recoExoticaValidationCaloHT"),
    recCaloJetLabel  = cms.InputTag("ak4CaloJets"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 
                                   500, 550, 600, 650, 700, 800, 900, 1000
                                   )
)
