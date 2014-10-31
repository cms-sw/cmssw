import FWCore.ParameterSet.Config as cms

DisplacedDimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoubleMu33NoFiltersNoVtx_v", # Run2
        "HLT_DoubleMu38NoFiltersNoVtx_v"  # Run2
        ),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings

    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                    60, 70, 80, 100
                                   ),
    )
