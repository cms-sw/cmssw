import FWCore.ParameterSet.Config as cms

MonojetBackupPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_CaloJet500_NoJetID_v", # Claimed path for Run3
        "HLT_CaloJet550_NoJetID_v", # Claimed path for Run3
        "HLT_PFJet500_v", # Claimed path for Run3
        "HLT_PFJet550_v" # Claimed path for Run3
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
