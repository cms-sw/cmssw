import FWCore.ParameterSet.Config as cms

JetNoBptxPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_JetE50_NoBPTX3BX_NoHalo_v" # Run2 proposal AND Run1 (frozenHLT)
        ),
    recCaloJetLabel    = cms.InputTag("ak5CaloJets"),

    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),

    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                                    160, 170, 180, 190, 200,
                                    220, 240, 260, 280, 300, 
                                    320, 340, 360, 380, 400,
                                    420, 440, 460, 480, 500),
    )
