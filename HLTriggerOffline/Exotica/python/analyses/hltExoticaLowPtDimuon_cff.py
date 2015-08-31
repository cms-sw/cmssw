import FWCore.ParameterSet.Config as cms

LowPtDimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Mu30_TkMu11_v",  # Run2
        "HLT_Mu17_TkMu8_DZ_v",# Run2
        "HLT_Mu17_Mu8_DZ_v"  # Run2
        #"HLT_Mu17_Mu8_v",    # Run1
        #"HLT_Mu17_TkMu8_v",  # Run1
        #"HLT_Mu22_TkMu8_v"   # Run1
        ),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings

    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                    60, 70, 80, 100
                                   ),
    dropPt3 = cms.bool(True),
    )
