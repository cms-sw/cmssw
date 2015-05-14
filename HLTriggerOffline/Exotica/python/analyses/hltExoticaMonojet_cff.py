import FWCore.ParameterSet.Config as cms

MonojetPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        #"HLT_PFJet260_v", # Run2
        #"HLT_PFJetCen80_PFMETNoMu100_v",
        #"HLT_PFJetCen80_PFMHTNoPuNoMu100_v",
        #"HLT_PFCenJet140_PFMETNoMu100_PFMHTNoMu140_v",
        #"HLT_PFCenJet140_PFMETNoMu140_PFMHTNoMu140_v",
        #"HLT_PFCenJet150_PFMETNoMu150_PFMHTNoMu150_v",
        #"HLT_CaloJet500_NoID_v",
        #"HLT_CaloJet500_NoJetID_v",
        #"HLT_MonoCentralPFJet80_PFMETnoMu105_NHEF0p95_v" # Run1

        "HLT_PFMETNoMu90_NoiseCleaned_PFMHTNoMu90_IDTight_v", 
        "HLT_PFMETNoMu120_NoiseCleaned_PFMHTNoMu120_IDTight_v", 
        "HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_NoiseCleaned_v",
        "HLT_MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_NoiseCleaned_v",
        "HLT_CaloMET200_NoiseCleaned_v" 
    ),

    CaloJetLabel    = cms.InputTag("ak4CaloJets"),
    PFJetLabel      = cms.InputTag("ak4PFJets"),
    #GenJetLabel     = cms.InputTag("ak4GenJets"),
    PFMETLabel      = cms.InputTag("recoExoticaValidationMETNoMu"),
    PFMHTLabel      = cms.InputTag("recoExoticaValidationMHTNoMu"),
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
    )
