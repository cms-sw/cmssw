import FWCore.ParameterSet.Config as cms

MuonNoBptxPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_L2Mu20_NoVertex_3Sta_NoBPTX3BX_NoHalo_v", # Run2 proposal
        "HLT_L2Mu20_NoVertex_2Cha_NoBPTX3BX_NoHalo_v"  # Run1 frozenHLT
        ),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
                                    1100, 1200, 1500
                                   ),
    )
