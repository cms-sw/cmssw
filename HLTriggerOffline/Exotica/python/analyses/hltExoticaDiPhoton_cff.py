import FWCore.ParameterSet.Config as cms

DiPhotonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoublePhoton85_v",    # Run2 proposal # Claimed path for Run3
        "HLT_DoublePhoton70_v", # Claimed path for Run3
#        "HLT_DoublePhoton33_CaloIdL_v" # Not claimed path for Run3
        "HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto_Mass55_v", # Claimed path for Run3
        "HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto_v", # Claimed path for Run3
        "HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v", # Claimed path for Run3
        #"HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55_v" # Claimed path for Run3 but not existent anymore in CMSSW_12_3_X
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    #recPhotonLabel  = cms.InputTag("gedPhotons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                    60, 70, 80, 100, 110, 120, 130, 140, 150,
                                    160, 170, 180, 190, 200
                                  ),
    dropPt3 = cms.bool(True),
    )
