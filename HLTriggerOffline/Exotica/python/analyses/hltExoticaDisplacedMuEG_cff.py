import FWCore.ParameterSet.Config as cms

DisplacedMuEGPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Mu38NoFiltersNoVtx_Photon38_CaloIdL_v", # Run2 Displaced muons
        "HLT_Mu42NoFiltersNoVtx_Photon42_CaloIdL_v", # Run2 Displaced muons
        #"HLT_Mu33NoFilters_NoVtx_2Cha_Photon33_CaloIdL_R9Id65_HE10_v", # Run2 
        "HLT_Mu33NoFiltersNoVtx_Photon33_CaloIdL_R9Id65_HE10_v", # Run2 
        "HLT_Mu28NoFiltersNoVtxDisplaced_Photon28_CaloIdL",
        "HLT_Mu33NoFiltersNoVtxDisplaced_Photon33_CaloIdL"
        #"HLT_Photon36_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon22_AND_HE10_R9Id65_Eta2_Mass15_v1", 
        #"HLT_Photon26_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon16_AND_HE10_R9Id65_Eta2_Mass60_v1"
        #"HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v", # Run1
        #"HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v"  # Run1
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    #recPhotonLabel  = cms.InputTag("gedPhotons"),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( #0, 5, 10, 15, 20,
                                    #25, 30, 35, 40,
                                    #45, 50, 55, 60,
                                    #65, 70, 75, 80,
                                    #85, 90, 95, 100,
                                    #105, 110, 115, 120,
                                    #125, 130, 135, 140,
                                    #145, 150, 155, 160,
                                    #165, 170, 175, 180,
                                    #185, 190, 195, 200),
                                    0, 10, 20, 30, 40, 50,
                                    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600
    )
)
