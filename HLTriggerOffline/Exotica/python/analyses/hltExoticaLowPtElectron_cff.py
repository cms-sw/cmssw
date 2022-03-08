import FWCore.ParameterSet.Config as cms

LowPtElectronPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
#        "HLT_Ele27_WP85_Gsf_v" # Run2 proposal
        #"HLT_Ele27_WP80_v"    # Run1 
        "HLT_Ele28_WPTight_Gsf_v",
        "HLT_Ele32_WPTight_Gsf_L1DoubleEG_v",
        "HLT_Ele32_WPTight_Gsf_v",
        "HLT_Ele35_WPTight_Gsf_v"
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                    60, 70, 80, 100
                                   ),
    dropPt2 = cms.bool(True),
    dropPt3 = cms.bool(True),
    )
