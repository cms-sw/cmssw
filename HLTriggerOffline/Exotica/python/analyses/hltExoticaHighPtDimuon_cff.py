import FWCore.ParameterSet.Config as cms

HighPtDimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Mu30_TkMu11_v", # Run2
        "HLT_Mu40_TkMu11_v", # Run2, backup
        "HLT_Mu27_TkMu8_v", # Run2, control
        "HLT_Mu17_Mu8_v"   # Run2 & Run1
        #"HLT_Mu17_TkMu8_v", # Run1
        #"HLT_Mu22_TkMu8_v"  # Run1
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
