import FWCore.ParameterSet.Config as cms

PFHTPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v",
        "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v",
        "HLT_PFHT750_4Jet_v",
        "HLT_PFHT550_4JetPt50_v",
        "HLT_PFHT650_4JetPt50_v",
        "HLT_PFHT750_4JetPt50_v",
        "HLT_PFHT650_4Jet_v", # Run2
        "HLT_PFHT550_4Jet_v", # Run2
        "HLT_PFHT800_v",
        "HLT_PFHT650_v",
        "DST_HT450_PFReco_PFBTagCSVReco_PFScouting_v",
        "DST_L1HTT125ORHTT150ORHTT175_PFReco_PFBTagCSVReco_PFScouting_v",
        "DST_CaloJet40_PFReco_PFBTagCSVReco_PFScouting_v"
        ),
    recPFMHTLabel  = cms.InputTag("recoExoticaValidationHT"),
    recPFJetLabel  = cms.InputTag("ak4PFJets"),
    # -- Analysis specific cuts
    #MET_genCut      = cms.string("sumEt > 75"),
    #MET_recCut      = cms.string("sumEt > 75"),
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 
                                   500, 550, 600, 650, 700, 800, 900, 1000
                                   )
)
