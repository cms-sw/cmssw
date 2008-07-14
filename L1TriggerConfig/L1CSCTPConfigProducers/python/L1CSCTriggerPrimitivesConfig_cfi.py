import FWCore.ParameterSet.Config as cms

l1csctpconf = cms.ESProducer("L1CSCTriggerPrimitivesConfigProducer",

    # Defines the set of parameters used in MTCC.
    isMTCC = cms.bool(False),

    # Defines the set of parameters used in the minus-endcap slice test with
    # the new TMB07 firmware.
    isTMB07 = cms.bool(True),

    # Parameters for ALCT processors: default
    alctParam = cms.PSet(
        alctFifoTbins   = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10),
        alctBxWidth     = cms.uint32(6),
        alctDriftDelay  = cms.uint32(3),
        alctNphThresh   = cms.uint32(2),
        alctNphPattern  = cms.uint32(4),
        alctAccThresh   = cms.uint32(2),
        alctAccPattern  = cms.uint32(4),
        alctTrigMode    = cms.uint32(3),
        alctAlctAmode   = cms.uint32(1),
        alctL1aWindow   = cms.uint32(5)
    ),

    # Parameters for ALCT processors: MTCC-II and 2007 tests of new firmware
    alctParamMTCC2 = cms.PSet(
        alctFifoTbins   = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10),
        alctBxWidth     = cms.uint32(6),
        alctDriftDelay  = cms.uint32(2),
        alctNphThresh   = cms.uint32(2),
        alctNphPattern  = cms.uint32(4),
        alctAccThresh   = cms.uint32(2),
        alctAccPattern  = cms.uint32(4),
        alctTrigMode    = cms.uint32(2),
        alctAlctAmode   = cms.uint32(0),
        alctL1aWindow   = cms.uint32(7)
    ),

    # Parameters for CLCT processors: default and 2007 tests of new firmware
    clctParam = cms.PSet(
        clctFifoTbins   = cms.uint32(12),
        clctFifoPretrig = cms.uint32(7),
        clctBxWidth     = cms.uint32(6),
        clctDriftDelay  = cms.uint32(2),
        clctHsThresh    = cms.uint32(2),
        clctDsThresh    = cms.uint32(2),
        clctNphPattern  = cms.uint32(4),
        # TMB07 parameters
        clctHitThresh   = cms.uint32(2),
        clctPidThresh   = cms.uint32(2),
        clctSepSrc      = cms.uint32(1),
        clctSepVme      = cms.uint32(10)
    ),

    # Parameters for CLCT processors: MTCC-II
    clctParamMTCC2 = cms.PSet(
        clctFifoTbins   = cms.uint32(12),
        clctFifoPretrig = cms.uint32(7),
        clctBxWidth     = cms.uint32(6),
        clctDriftDelay  = cms.uint32(2),
        clctHsThresh    = cms.uint32(4),
        clctDsThresh    = cms.uint32(4),
        clctNphPattern  = cms.uint32(1),
        # TMB07 parameters
        clctHitThresh   = cms.uint32(2),
        clctPidThresh   = cms.uint32(2),
        clctSepSrc      = cms.uint32(1),
        clctSepVme      = cms.uint32(10)
    )
)
