import FWCore.ParameterSet.Config as cms

HighPtPhotonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Photon175_v",  # Run2 proposal
        "HLT_Photon165_HE10_v",  # Run2 proposal
        "HLT_Photon36_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon31_AND_HE10_R9Id65_Mass10_v",  # Run2 proposal
        "HLT_Photon26_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon16_AND_HE10_R9Id65_Mass60_v",  # Run2 proposal
        "HLT_Photon90_CaloIdL_PFHT600_v" #50ns backup menu
        #"HLT_Photon135_v"  # Run1 (frozenHLT)
        ),
    recPhotonLabel  = cms.InputTag("gedPhotons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 50, 100, 150, 200, 
                                    250, 300, 400, 500, 600, 700, 800, 900, 1000,
                                    1100, 1200, 1500
                                   ),
    )
