import FWCore.ParameterSet.Config as cms

DSTMuonsPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "DST_ZeroBias_CaloScouting_PFScouting_v",
        "DST_ZeroBias_BTagScouting_v",
        "DST_L1DoubleMu_CaloScouting_PFScouting_v",
        "DST_L1DoubleMu_BTagScouting_v",
        "DST_DoubleMu3_Mass10_CaloScouting_PFScouting_v",
        "DST_DoubleMu3_Mass10_BTagScouting_v",
        "HLT_DoubleMu3_Mass10_v"
        ),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    #MET_genCut      = cms.string("sumEt > 75"),
    #MET_recCut      = cms.string("sumEt > 75"),
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 
                                   500, 550, 600, 650, 700, 800, 900, 1000
                                   ),
    dropPt3 = cms.bool(True),
)
