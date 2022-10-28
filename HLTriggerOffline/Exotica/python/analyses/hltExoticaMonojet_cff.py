import FWCore.ParameterSet.Config as cms

MonojetPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", # Claimed path for Run3
#        "HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v", # Not claimed path for Run3
        "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_v", # Claimed path for Run3
#        "HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v", # Not claimed path for Run3
#        "HLT_PFMET110_PFMHT110_IDTight_v", # Not claimed path for Run3
        "HLT_PFMET120_PFMHT120_IDTight_v", # Claimed path for Run3
#        "HLT_PFMET130_PFMHT130_IDTight_v", # Not claimed path for Run3
#        "HLT_PFMET140_PFMHT140_IDTight_v", # Not claimed path for Run3
#        "HLT_PFMETTypeOne110_PFMHT110_IDTight_v", # Not claimed path for Run3
#        "HLT_PFMETTypeOne120_PFMHT120_IDTight_v", # Not claimed path for Run3
#        "HLT_PFMETTypeOne130_PFMHT130_IDTight_v", # Not claimed path for Run3
#        "HLT_PFMETTypeOne140_PFMHT140_IDTight_v", # Not claimed path for Run3
        "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", # Claimed path for Run3
        "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", # Claimed path for Run3
#        "HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight_v", # Not claimed path for Run3
#        "HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight_v", # Not claimed path for Run3
        "HLT_PFHT500_PFMET100_PFMHT100_IDTight_v", # Claimed path for Run3
        "HLT_PFHT500_PFMET110_PFMHT110_IDTight_v", # Claimed path for Run3
        "HLT_PFHT700_PFMET85_PFMHT85_IDTight_v", # Claimed path for Run3
        "HLT_PFHT700_PFMET95_PFMHT95_IDTight_v", # Claimed path for Run3
        "HLT_PFHT800_PFMET75_PFMHT75_IDTight_v", # Claimed path for Run3
        "HLT_PFHT800_PFMET85_PFMHT85_IDTight_v", # Claimed path for Run3
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
                                    160, 170, 180, 190, 200, 220, 240, 260, 280, 300,
                                    320, 340, 360, 380, 400, 420, 440, 460, 480, 500,
                                    600, 700, 800, 900, 1100, 1200, 1400,1500
                                  ),

    parametersTurnOnSumEt = cms.vdouble(    0,  100,  200,  300,  400,  500,  600,  700,  800,  900,
                                         1000, 1100, 1200, 1300, 1400, 1500
                                       ),

    dropPt2 = cms.bool(True),
    dropPt3 = cms.bool(True),
    )
