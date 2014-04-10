import FWCore.ParameterSet.Config as cms

HighPtDielectronPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v",
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                    60, 70, 80, 100, 120, 140, 160, 180, 200
                                   ),
    )
