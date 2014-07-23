import FWCore.ParameterSet.Config as cms

HighPtPhotonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Photon80_v",
        ),
    recPhotonLabel  = cms.InputTag("gedPhotons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
                                    1100, 1200, 1500
                                   ),
    )
