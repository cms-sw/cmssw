import FWCore.ParameterSet.Config as cms

DisplacedDimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoubleMu43NoFiltersNoVtx_v", # 2017 displaced mu-mu (main) # Claimed path for Run3
        "HLT_DoubleL3Mu16_10NoVtx_DxyMin0p01cm_v", #New Run3 path (introduced in HLT V1.3)
        "HLT_DoubleL3dTksMu16_10NoVtx_DxyMin0p01cm_v", #New Run3 path (introduced in HLT V1.3)
        ),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings
    parametersDxy      = cms.vdouble(50, -50, 50),
    parametersTurnOn = cms.vdouble(
                                    0, 10, 20, 30, 40, 50,
                                    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600
                                   ),
    dropPt3 = cms.bool(True),
    )
