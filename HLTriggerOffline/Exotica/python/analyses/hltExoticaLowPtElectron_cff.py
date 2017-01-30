import FWCore.ParameterSet.Config as cms

LowPtElectronPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_Ele27_WP85_Gsf_v" # Run2 proposal
        #"HLT_Ele27_WP80_v"    # Run1 
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    #parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    #                                60, 70, 80, 100, 120, 140, 160, 180, 200
    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                    60, 70, 80, 100
                                   ),
    dropPt3 = cms.bool(True),
    )
