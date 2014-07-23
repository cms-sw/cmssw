import FWCore.ParameterSet.Config as cms

HighPtDimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        #"HLT_Mu17_Mu8_v",
        #"HLT_Mu17_TkMu8_v",
        "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v",
        "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
        "HLT_Mu30_TkMu11_v",
        ),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                    60, 70, 80, 100, 120, 140, 160, 180, 200,
                                    220, 240, 260, 280, 300,
                                    320, 340, 360, 380, 400,
                                    420, 440, 460, 480, 500,
                                   ),
    )
