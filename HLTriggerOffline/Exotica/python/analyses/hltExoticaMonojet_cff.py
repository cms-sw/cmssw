import FWCore.ParameterSet.Config as cms

MonojetPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        #"HLT_PFJet260_v", # Run2
        #"HLT_PFJetCen80_PFMETNoMu100_v",
        #"HLT_PFJetCen80_PFMHTNoPuNoMu100_v",
        #"HLT_PFCenJet140_PFMETNoMu100_PFMHTNoMu140_v",
        #"HLT_PFCenJet140_PFMETNoMu140_PFMHTNoMu140_v",
        #"HLT_PFCenJet150_PFMETNoMu150_PFMHTNoMu150_v",
        "HLT_MonoCentralPFJet140_PFMETNoMu100_PFMHTNoMu140_NoiseCleaned_v", 
        "HLT_MonoCentralPFJet140_PFMETNoMu140_PFMHTNoMu140_NoiseCleaned_v",
        "HLT_MonoCentralPFJet150_PFMETNoMu150_PFMHTNoMu150_NoiseCleaned_v"
        #"HLT_CaloJet500_NoID_v",
        #"HLT_CaloJet500_NoJetID_v",
        #"HLT_MonoCentralPFJet80_PFMETnoMu105_NHEF0p95_v" # Run1
        ),

    recCaloJetLabel  = cms.InputTag("ak4CaloJets"),
    recPFJetLabel    = cms.InputTag("ak4PFJets"),
    recPFMETLabel    = cms.InputTag("recoExoticaValidationMETNoMu"),
    recPFMHTLabel    = cms.InputTag("recoExoticaValidationMHTNoMu"),

    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                                    160, 170, 180, 190, 200,
                                    220, 240, 260, 280, 300, 
                                    320, 340, 360, 380, 400,
                                    420, 440, 460, 480, 500),
    )
