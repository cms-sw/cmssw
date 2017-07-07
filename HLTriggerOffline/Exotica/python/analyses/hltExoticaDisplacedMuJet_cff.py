import FWCore.ParameterSet.Config as cms

DisplacedMuJetPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        # Signal
        #"HLT_Mu33NoFiltersNoVtxDisplaced_DisplacedJet50_Loose_v",
        #"HLT_Mu33NoFiltersNoVtxDisplaced_DisplacedJet50_Tight_v",
        "HLT_Mu38NoFiltersNoVtx_DisplacedJet60_Loose_v",
        # Backup
        "HLT_Mu38NoFiltersNoVtxDisplaced_DisplacedJet60_Loose_v",
        "HLT_Mu38NoFiltersNoVtxDisplaced_DisplacedJet60_Tight_v",
        # Control
        "HLT_Mu28NoFiltersNoVtx_DisplacedJet40_Loose_v",
        "HLT_Mu28NoFiltersNoVtx_CentralCaloJet40_v",

        "HLT_Mu23NoFiltersNoVtx_Photon23_CaloIdL_v",
        "HLT_DoubleMu18NoFiltersNoVtx_v"
        ),
    recMuonLabel  = cms.InputTag("muons"),
    recPFJetLabel = cms.InputTag("ak4PFJets"),
    #recPFJetLabel = cms.InputTag("hltAK4PFJetsForTaus"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings

    parametersDxy      = cms.vdouble(50, -50, 50),
    parametersTurnOn = cms.vdouble( 
                                    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600
                                   ),
    dropPt3 = cms.bool(True),
    )
