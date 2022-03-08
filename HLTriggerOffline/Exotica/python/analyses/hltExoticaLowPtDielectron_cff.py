import FWCore.ParameterSet.Config as cms

LowPtDielectronPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
#        "HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_MW_v", # Run2
#        "HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v",    # Run1 & Run2
#        "HLT_DoubleEle33_CaloIdL_MW_v",            # 2016 menu # Not claimed path for Run3
#        "HLT_DoubleEle33_CaloIdL_v"                # 2016 menu
        "HLT_DoubleEle25_CaloIdL_MW_v",
        "HLT_DoubleEle27_CaloIdL_MW_v",
        "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v",
        "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v"
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings

    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                    60, 70, 80, 100
                                   ),
    dropPt3 = cms.bool(True),
    )
