import FWCore.ParameterSet.Config as cms

DiPhotonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoublePho80_v",    # Run2 proposal
        "HLT_DoublePhoton70_v"  # Run1 (frozenHLT)
        ),
    recPhotonLabel  = cms.InputTag("gedPhotons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
                                    1100, 1200, 1500
                                   ),
    )
