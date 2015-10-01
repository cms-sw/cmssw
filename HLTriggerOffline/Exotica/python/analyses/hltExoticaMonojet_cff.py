import FWCore.ParameterSet.Config as cms

MonojetPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_PFMETNoMu90_PFMHTNoMu90_IDTight_v",
        "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v",
        "HLT_MET200_v",
        "HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v",
        "HLT_MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_IDTight_v",

        # For backward compatibility
        "HLT_PFMETNoMu90_JetIdCleaned_PFMHTNoMu90_IDTight_v",
        "HLT_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v",
        "HLT_MET200_JetIdCleaned_v",
        "HLT_MonoCentralPFJet80_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v",
        "HLT_MonoCentralPFJet80_PFMETNoMu90_JetIdCleaned_PFMHTNoMu90_IDTight_v"
    ),

    recCaloJetLabel    = cms.InputTag("ak4CaloJets"),
    recPFJetLabel      = cms.InputTag("ak4PFJets"),
    #GenJetLabel     = cms.InputTag("ak4GenJets"),
    recPFMETLabel      = cms.InputTag("recoExoticaValidationMETNoMu"),
    recPFMHTLabel      = cms.InputTag("recoExoticaValidationMHTNoMu"),
    #PFMETLabel      = cms.InputTag("pfMet"),
    #PFMHTLabel      = cms.InputTag("recoExoticaValidationHT"),

    #GenMETLabel     = cms.InputTag("genMetTrue"),
    #GenMETCaloLabel = cms.InputTag("genMetCalo"),

    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                                    160, 170, 180, 190, 200,
                                    220, 240, 260, 280, 300, 
                                    320, 340, 360, 380, 400,
                                    420, 440, 460, 480, 500,600,700,800,900,1100,1200,
                                    1400,1600,1800,2000,2200,2400,2600,2800,3000),

    parametersTurnOnSumEt = cms.vdouble(    0,  100,  200,  300,  400,  500,  600,  700,  800,  900,
                                         1000, 1100, 1200, 1300, 1400, 1500
                                       ),

    dropPt2 = cms.bool(True),
    dropPt3 = cms.bool(True),
    )
