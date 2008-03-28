import FWCore.ParameterSet.Config as cms

l1csctpconf = cms.ESProducer("L1CSCTriggerPrimitivesConfigProducer",
    # Parameters for ALCT processors: default
    alctParam = cms.PSet(
        alctL1aWindow = cms.uint32(5),
        alctTrigMode = cms.uint32(3),
        alctNphPattern = cms.uint32(4),
        alctBxWidth = cms.uint32(6),
        alctAlctAmode = cms.uint32(1),
        alctDriftDelay = cms.uint32(3),
        alctNphThresh = cms.uint32(2),
        alctFifoTbins = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10)
    ),
    # Defines the set of parameters used in the minus-endcap slice test with
    # the new TMB07 firmware.
    isTMB07 = cms.bool(True),
    # Parameters for CLCT processors: MTCC-II
    clctParamMTCC2 = cms.PSet(
        clctFifoPretrig = cms.uint32(7),
        clctDriftDelay = cms.uint32(2),
        clctFifoTbins = cms.uint32(12),
        clctNphPattern = cms.uint32(1),
        clctBxWidth = cms.uint32(6),
        clctHsThresh = cms.uint32(4),
        clctHitThresh = cms.uint32(2),
        clctSepSrc = cms.uint32(1),
        clctSepVme = cms.uint32(10),
        clctDsThresh = cms.uint32(4),
        clctPidThresh = cms.uint32(2)
    ),
    # Defines the set of parameters used in MTCC.
    isMTCC = cms.bool(False),
    # Parameters for CLCT processors: default and 2007 tests of new firmware
    clctParam = cms.PSet(
        clctFifoPretrig = cms.uint32(7),
        clctDriftDelay = cms.uint32(2),
        clctFifoTbins = cms.uint32(12),
        clctNphPattern = cms.uint32(4),
        clctBxWidth = cms.uint32(6),
        clctHsThresh = cms.uint32(2),
        clctHitThresh = cms.uint32(2),
        clctSepSrc = cms.uint32(1),
        clctSepVme = cms.uint32(10),
        clctDsThresh = cms.uint32(2),
        clctPidThresh = cms.uint32(2)
    ),
    # Parameters for ALCT processors: MTCC-II and 2007 tests of new firmware
    alctParamMTCC2 = cms.PSet(
        alctL1aWindow = cms.uint32(3),
        alctTrigMode = cms.uint32(2),
        alctNphPattern = cms.uint32(4),
        alctBxWidth = cms.uint32(6),
        alctAlctAmode = cms.uint32(0),
        alctDriftDelay = cms.uint32(3),
        alctNphThresh = cms.uint32(2),
        alctFifoTbins = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10)
    )
)


