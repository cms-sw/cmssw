import FWCore.ParameterSet.Config as cms

DiPhotonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoublePho85_v",    # Run2 proposal
        "HLT_Photon36_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon22_AND_HE10_R9Id65_Eta2_Mass15_v",
        "HLT_Photon26_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon16_AND_HE10_R9Id65_Eta2_Mass60_v",
        "HLT_Photon42_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon25_AND_HE10_R9Id65_Eta2_Mass15_v", #50ns backup menu
        "HLT_DoublePhoton60_v"
        #"HLT_DoublePhoton70_v"  # Run1 (frozenHLT)
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    #recPhotonLabel  = cms.InputTag("gedPhotons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
                                    1100, 1200, 1500
                                   ),
    )
