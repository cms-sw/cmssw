import FWCore.ParameterSet.Config as cms

DisplacedMuEGPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v", # 2017 displaced e-mu (main)
        "HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL_v", # 2017 displaced e-mu (backup)
        "HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_v", # 2017 displaced e-mu, muon with dxy> 0.01cm (main)
        "HLT_Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_v", # 2017 displaced e-mu, muon with dxy> 0.01cm (backup)
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersDxy      = cms.vdouble(50, -50, 50),
    parametersTurnOn = cms.vdouble(
                                    0, 10, 20, 30, 40, 50,
                                    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600
    ),
    dropPt2 = cms.bool(True),
    dropPt3 = cms.bool(True),
)
