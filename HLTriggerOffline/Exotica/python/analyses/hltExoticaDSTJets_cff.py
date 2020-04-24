import FWCore.ParameterSet.Config as cms

DSTJetsPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "DST_CaloJet40_CaloScouting_PFScouting_v",
        "DST_CaloJet40_BTagScouting_v",
        "DST_L1HTT_CaloScouting_PFScouting_v",
        "DST_L1HTT_BTagScouting_v",
        "DST_HT250_CaloScouting_v",
        "DST_HT410_PFScouting_v",
        "DST_HT410_BTagScouting_v",
        "DST_HT450_PFScouting_v",
        "DST_HT450_BTagScouting_v",
        # 2016 menu
        "DST_HT250_CaloBTagScouting_v",
        "DST_L1HTT_CaloBTagScouting_v",
        "DST_CaloJet40_CaloBTagScouting_v",
        # For backward compatibility
        "DST_HT250_CaloScouting_v", 
        "DST_CaloJet40_CaloScouting_v", 
        "DST_L1HTT125ORHTT150ORHTT175_CaloScouting_v",
        "DST_HT450_PFReco_PFBTagCSVReco_PFScouting_v",
        "DST_L1HTT125ORHTT150ORHTT175_PFReco_PFBTagCSVReco_PFScouting_v",
        "DST_CaloJet40_PFReco_PFBTagCSVReco_PFScouting_v"
        ),
    recPFMHTLabel  = cms.InputTag("recoExoticaValidationHT"),
    recPFJetLabel  = cms.InputTag("ak4PFJets"),
    recCaloMHTLabel  = cms.InputTag("recoExoticaValidationCaloHT"),
    recCaloJetLabel  = cms.InputTag("ak4CaloJets"),
    # -- Analysis specific cuts
    #MET_genCut      = cms.string("sumEt > 75"),
    #MET_recCut      = cms.string("sumEt > 75"),
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 
                                   500, 550, 600, 650, 700, 800, 900, 1000
                                   )
)
