import FWCore.ParameterSet.Config as cms

DisplacedEleMuPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Mu38NoFiltersNoVtx_Photon38_CaloIdL_v", # Run2 Displaced muons
        "HLT_Mu42NoFiltersNoVtx_Photon42_CaloIdL_v", # Run2 Displaced muons
        "HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v", # Run1
        "HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v"  # Run1
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20,
                                    25, 30, 35, 40,
                                    45, 50, 55, 60,
                                    65, 70, 75, 80,
                                    85, 90, 95, 100,
                                    105, 110, 115, 120,
                                    125, 130, 135, 140,
                                    145, 150, 155, 160,
                                    165, 170, 175, 180,
                                    185, 190, 195, 200),
    dropPt2 = cms.bool(True),
    dropPt3 = cms.bool(True),
    )
