import FWCore.ParameterSet.Config as cms

HighPtPhotonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Photon175_v",  # Run2 proposal # Claimed path for Run3
#        "HLT_Photon165_HE10_v",  # Run2 proposal
#        "HLT_Photon36_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon31_AND_HE10_R9Id65_Mass10_v",  # Run2 proposal
#        "HLT_Photon26_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon16_AND_HE10_R9Id65_Mass60_v",  # Run2 proposal
#        "HLT_Photon90_CaloIdL_PFHT500_v", #50ns backup menu
#        "HLT_Photon150_v", # 0T # Not claimed path for Run3
        #"HLT_Photon135_v"  # Run1 (frozenHLT)
        "HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15_v", # 2017 # Claimed path for Run3
#        "HLT_Photon33_v", # 2017 # Not claimed path for Run3
#        "HLT_Photon60_R9Id90_CaloIdL_IsoL_v", # 2017 # Not claimed path for Run3
#        "HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_v" # 2017 # Not claimed path for Run3
        "HLT_Photon110EB_TightID_TightIso_v", 
        "HLT_Photon200_v"
        ),
    recPhotonLabel  = cms.InputTag("gedPhotons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 25, 50, 75, 100, 125, 150, 175, 200, 225,
                                    250, 275, 300, 400, 500, 600, 700, 800, 900, 1000
                                   ),
    dropPt2 = cms.bool(True),
    dropPt3 = cms.bool(True),
    )
