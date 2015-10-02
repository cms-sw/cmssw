import FWCore.ParameterSet.Config as cms

MonojetBackupPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        #"HLT_PFJet260_v", # Run2
        #"HLT_PFJetCen80_PFMETNoMu100_v",
        #"HLT_PFJetCen80_PFMHTNoPuNoMu100_v",
        #"HLT_PFCenJet140_PFMETNoMu100_PFMHTNoMu140_v",
        #"HLT_PFCenJet140_PFMETNoMu140_PFMHTNoMu140_v",
        #"HLT_PFCenJet150_PFMETNoMu150_PFMHTNoMu150_v",
        #"HLT_CaloJet500_NoID_v",
        "HLT_CaloJet500_NoJetID_v"
        #"HLT_MonoCentralPFJet80_PFMETnoMu105_NHEF0p95_v" # Run1
        ),
    recPFJetLabel    = cms.InputTag("ak4PFJets"),
    recPFMETLabel    = cms.InputTag("pfMet"),
    recCaloJetLabel  = cms.InputTag("ak4CaloJets"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 
                                    400, 410, 420, 430, 440, 450, 460, 470, 480, 490,
                                    500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600
                                  ),
    dropPt2 = cms.bool(True),
    dropPt3 = cms.bool(True),
    )
