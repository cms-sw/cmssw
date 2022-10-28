import FWCore.ParameterSet.Config as cms

HighPtDimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        ),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
                                    1100, 1200, 1500
                                   ),
    dropPt3 = cms.bool(True),
    )
