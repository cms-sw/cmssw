import FWCore.ParameterSet.Config as cms

HighPtElectronPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Ele95_CaloIdVT_GsfTrkIdT_v", # Run2 proposal
        "HLT_Ele80_CaloIdVT_GsfTrkIdT_v", # Run1
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
                                    1100, 1200, 1500
                                   ),
    )
