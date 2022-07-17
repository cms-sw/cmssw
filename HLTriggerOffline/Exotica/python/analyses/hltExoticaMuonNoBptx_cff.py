import FWCore.ParameterSet.Config as cms

MuonNoBptxPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
#        "HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v", # 2017 proposal # Claimed path for Run3, but a backup so no need to monitor it closely here
#        "HLT_L2Mu10_NoVertex_NoBPTX_v", # Claimed path for Run3, but a control path so no need to monitor it closely here
#        "HLT_L2Mu10_NoVertex_NoBPTX3BX_v", # Claimed path for Run3, but a control path so no need to monitor it closely here
        "HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v" # Claimed path for Run3
        ),
    #recMuonLabel  = cms.InputTag("muons"),
    recMuonTrkLabel  = cms.InputTag("displacedStandAloneMuons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                    60, 70, 80, 100
                                   ),
    dropPt2 = cms.bool(True),
    dropPt3 = cms.bool(True),
    )
