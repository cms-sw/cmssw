import FWCore.ParameterSet.Config as cms

DisplacedDimuonDijetPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_DoubleDisplacedMu8NoFilters_DisplacedDijet60_v"
        ),
    recMuonLabel  = cms.InputTag("muons"),
    recPFJetLabel = cms.InputTag("ak4PFJets"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings

    parametersDxy      = cms.vdouble(50, -50, 50),
    parametersTurnOn = cms.vdouble( 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                    60, 70, 80, 100
                                   ),
    )
