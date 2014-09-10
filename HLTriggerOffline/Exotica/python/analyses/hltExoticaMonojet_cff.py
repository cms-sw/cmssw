import FWCore.ParameterSet.Config as cms

MonojetPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_MonoCentralPFJet80_PFMETnoMu105_NHEF0p95_v"
        ),
    recJetLabel    = cms.InputTag("ak5PFJetsCHS"),
    recPFMETLabel  = cms.InputTag("pfMet"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(2),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                                    160, 170, 180, 190, 200,
                                    220, 240, 260, 280, 300, 
                                    320, 340, 360, 380, 400,
                                    420, 440, 460, 480, 500),
    )
