import FWCore.ParameterSet.Config as cms

SingleMuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Mu45_eta2p1_v", # Run 2 
        "HLT_Mu50_v", # Run 2
        #50ns backup menu
        "HLT_Mu55_v",
        "HLT_Mu50_eta2p1_v"
        ),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 
                                    42, 44, 46, 48, 50, 52, 54, 56, 58, 60,
                                    70, 80, 90, 100
                                   ),
    dropPt2 = cms.bool(True),
    dropPt3 = cms.bool(True),
    )
