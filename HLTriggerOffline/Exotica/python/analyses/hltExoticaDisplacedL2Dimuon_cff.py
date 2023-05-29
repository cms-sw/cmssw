import FWCore.ParameterSet.Config as cms

DisplacedL2DimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_v", #Claimed for Run 3
        "HLT_DoubleL2Mu23NoVtx_2Cha_v", #Claimed for Run 3
        "HLT_DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v", #New for Run 3 (introduced in HLT V1.3)
        "HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm_v", #New for Run 3 (introduced in HLT V1.3)
        "HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v", #New for Run 3 (introduced in HLT V1.3)
        ),
    recMuonLabel  = cms.InputTag("muons"),
    recMuonTrkLabel  = cms.InputTag("displacedStandAloneMuons"),

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
