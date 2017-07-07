import FWCore.ParameterSet.Config as cms

DisplacedDimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoubleMu33NoFiltersNoVtx_v", # Run2
        "HLT_DoubleMu38NoFiltersNoVtx_v", # Run2
        "HLT_DoubleMu23NoFiltersNoVtxDisplaced_v", # Run2
        "HLT_DoubleMu28NoFiltersNoVtxDisplaced_v" # Run2
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
