import FWCore.ParameterSet.Config as cms

DisplacedDiPhotonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DiPhoton10Time1p4ns_v", # New for Run3 (introduced in HLT V1.3)
        "HLT_DiPhoton10sminlt0p1_v", # New for Run3 (introduced in HLT V1.3)
        ),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 25, 50, 75, 100, 125, 150, 175, 200, 225,
                                    250, 275, 300, 400, 500, 600, 700, 800, 900, 1000
                                   ),
    parametersDxy      = cms.vdouble(50, -50, 50),
    dropPt3 = cms.bool(True),
    )
