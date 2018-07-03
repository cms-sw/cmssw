import FWCore.ParameterSet.Config as cms

PureMETPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_PFMET170_NotCleaned_v",
        "HLT_PFMET170_v",
        "HLT_PFMET170_HBHECleaned_v",
        "HLT_PFMET170_v",
        "HLT_PFMET170_NoiseCleaned_v",  # Run2
        "HLT_MET200_v",
        "HLT_MET100_v",    # 0T
        "HLT_MET150_v",    # 0T

        # For backward compatibility
        "HLT_PFMET170_JetIdCleaned_v",  
        "HLT_MET200_JetIdCleaned_v",
        "HLT_MET100_JetIdCleaned_v",    # 0T
        "HLT_MET150_JetIdCleaned_v"     # 0T
        ),
    recPFMETLabel  = cms.InputTag("pfMet"),
    recCaloMETLabel = cms.InputTag("caloMet"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble( 0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                                    160, 170, 180, 190, 200,
                                    220, 240, 260, 280, 300, 
                                    320, 340, 360, 380, 400,
                                    420, 440, 460, 480, 500),
    )
