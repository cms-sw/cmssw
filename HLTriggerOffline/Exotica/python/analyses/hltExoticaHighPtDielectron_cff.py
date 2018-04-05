import FWCore.ParameterSet.Config as cms

HighPtDielectronPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoubleEle25_CaloIdL_GsfTrkIdVL_v", # 0T only
        "HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_MW_v", # Run2
        "HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v",    # Run1 & Run2
        "HLT_DoubleEle33_CaloIdL_MW_v",            # 2016 menu
        "HLT_DoubleEle33_CaloIdL_v"                # 2016 menu
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
                                    1100, 1200, 1500
                                   ),
    dropPt3 = cms.bool(True),
    )
