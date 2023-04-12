import FWCore.ParameterSet.Config as cms

HighPtPhotonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Photon175_v",  # Run2 proposal # Claimed path for Run3
        "HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350_v", # Path updated for 2023
        "HLT_Photon110EB_TightID_TightIso_v", # Claimed path for Run3 
        "HLT_Photon200_v" # Claimed path for Run3
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
