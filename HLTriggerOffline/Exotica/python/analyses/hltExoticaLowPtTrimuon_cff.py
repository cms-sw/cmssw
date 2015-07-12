import FWCore.ParameterSet.Config as cms

LowPtTrimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_TrkMu15_DoubleTrkMu5NoFiltersNoVtx_v", #signal
        "HLT_TrkMu17_DoubleTrkMu8NoFiltersNoVtx_v", #signal
        "HLT_DiMuon0_Jpsi_Muon",                    # control
        "HLT_Mu17_TkMu8_DZ",                        # backup
        "HLT_Mu17_Mu8_DZ"                           # backup
    ),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(3),
    # -- Analysis specific binnings
    parametersDxy      = cms.vdouble(50, -2.500, 2.500),
    )
