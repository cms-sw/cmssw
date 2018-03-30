import FWCore.ParameterSet.Config as cms

DisplacedDimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoubleMu43NoFiltersNoVtx_v", # 2017 displaced mu-mu (main)
        "HLT_DoubleMu48NoFiltersNoVtx_v", # 2017 displaced mu-mu (backup)
        "HLT_DoubleMu33NoFiltersNoVtxDisplaced_v", # 2017 displaced mu-mu, muons with dxy> 0.01 cm (main)
        "HLT_DoubleMu40NoFiltersNoVtxDisplaced_v", # 2017 displaced mu-mu, muons with dxy> 0.01 cm (backup)
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
