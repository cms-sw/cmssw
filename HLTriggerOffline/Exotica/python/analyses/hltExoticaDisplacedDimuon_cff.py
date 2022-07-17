import FWCore.ParameterSet.Config as cms

DisplacedDimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoubleMu43NoFiltersNoVtx_v", # 2017 displaced mu-mu (main) # Claimed path for Run3
#        "HLT_DoubleMu48NoFiltersNoVtx_v", # 2017 displaced mu-mu (backup) # Claimed path for Run3, but a backup so no need to monitor it closely here
#        "HLT_DoubleMu33NoFiltersNoVtxDisplaced_v", # 2017 displaced mu-mu, muons with dxy> 0.01 cm (main) # Claimed path for Run3, but being superseeded by HLT_DoubleL3Mu10NoVtx_Displaced_v)
#        "HLT_DoubleMu40NoFiltersNoVtxDisplaced_v", # 2017 displaced mu-mu, muons with dxy> 0.01 cm (backup) # Not claimed path for Run3
        "HLT_DoubleL3Mu10NoVtx_Displaced_v", #New Run3 path
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
