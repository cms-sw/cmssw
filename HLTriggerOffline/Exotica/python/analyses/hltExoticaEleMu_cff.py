import FWCore.ParameterSet.Config as cms

EleMuPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Mu30_Ele30_CaloIdL_GsfTrkIdVL_v"
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
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
    )
