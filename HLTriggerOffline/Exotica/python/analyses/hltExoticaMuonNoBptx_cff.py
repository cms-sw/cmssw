import FWCore.ParameterSet.Config as cms

MuonNoBptxPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_L2Mu35_NoVertex_3Sta_NoBPTX3BX_NoHalo_v", # Run2 proposal
        "HLT_L2Mu10_NoVertex_NoBPTX_v",
        "HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v",
        "HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_NoHalo_v"
        #"HLT_L2Mu20_NoVertex_2Cha_NoBPTX3BX_NoHalo_v"  # Run1 frozenHLT
        ),
    #recMuonLabel  = cms.InputTag("muons"),
    recMuonTrkLabel  = cms.InputTag("refittedStandAloneMuons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    #parametersTurnOn = cms.vdouble( 0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
    #                                1100, 1200, 1500
    #                               ),
    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                    60, 70, 80, 100
                                   ),
    dropPt2 = cms.bool(True),
    dropPt3 = cms.bool(True),
    )
