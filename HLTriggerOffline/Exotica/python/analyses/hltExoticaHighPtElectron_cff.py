import FWCore.ParameterSet.Config as cms

HighPtElectronPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Ele105_CaloIdVT_GsfTrkIdT_v", # Run2 proposal
        "HLT_Ele115_CaloIdVT_GsfTrkIdT_v"  # 50ns backup menu
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 50, 80, 90, 100, 110, 120, 150, 200, 250, 
                                    300, 400, 500, 600, 700, 800, 900, 1000,
                                    1100, 1200, 1500
                                   ),
    )
