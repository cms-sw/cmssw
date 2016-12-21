import FWCore.ParameterSet.Config as cms

DisplacedMuEGPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Mu38NoFiltersNoVtx_Photon38_CaloIdL_v", # Run2 Displaced muons
        "HLT_Mu42NoFiltersNoVtx_Photon42_CaloIdL_v", # Run2 Displaced muons
        "HLT_Mu33NoFiltersNoVtx_Photon33_CaloIdL_R9Id65_HE10_v", # Run2 
        "HLT_Mu28NoFiltersNoVtxDisplaced_Photon28_CaloIdL",
        "HLT_Mu33NoFiltersNoVtxDisplaced_Photon33_CaloIdL"
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
