import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
# Default parameters for CSCTriggerPrimitives generator
# =====================================================
cscTriggerPrimitiveDigis = cms.EDProducer("CSCTriggerPrimitivesProducer",
    CSCCommonTrigger,

    # Name of digi producer module(s)
    CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),

    # Parameters common for all boards
    commonParam = cms.PSet(
        isTMB07 = cms.bool(True),
        isMTCC = cms.bool(True)
    ),

    # Parameters for ALCT processors: default
    alctParamDef = cms.PSet(
        alctFifoTbins   = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10),
        alctBxWidth     = cms.uint32(6),
        alctDriftDelay  = cms.uint32(3),
        alctNphThresh   = cms.uint32(2),
        alctNphPattern  = cms.uint32(4),
        alctAccThresh   = cms.uint32(2),
        alctAccPattern  = cms.uint32(4),
        alctTrigMode    = cms.uint32(3),
        alctMode        = cms.uint32(1),
        alctL1aWindow   = cms.uint32(5),
        verbosity = cms.untracked.int32(0)
    ),

    # Parameters for ALCT processors: MTCC-I
    alctParamMTCC1 = cms.PSet(
        alctFifoTbins   = cms.uint32(10),
        alctFifoPretrig = cms.uint32(12),
        alctBxWidth     = cms.uint32(6),
        alctDriftDelay  = cms.uint32(3),
        alctNphThresh   = cms.uint32(4),
        alctNphPattern  = cms.uint32(4),
        alctAccThresh   = cms.uint32(4),
        alctAccPattern  = cms.uint32(4),
        alctTrigMode    = cms.uint32(0),
        alctMode        = cms.uint32(2),
        alctL1aWindow   = cms.uint32(3),
        verbosity = cms.untracked.int32(0)
    ),

    # Parameters for ALCT processors: MTCC-II
    alctParamMTCC2 = cms.PSet(
        alctFifoTbins   = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10),
        alctBxWidth     = cms.uint32(6),
        alctDriftDelay  = cms.uint32(3),
        alctNphThresh   = cms.uint32(2),
        alctNphPattern  = cms.uint32(4),
        alctAccThresh   = cms.uint32(2),
        alctAccPattern  = cms.uint32(4),
        alctTrigMode    = cms.uint32(2),
        alctMode        = cms.uint32(0),
        alctL1aWindow   = cms.uint32(3),
        verbosity = cms.untracked.int32(0)
    ),

    # Parameters for CLCT processors: default
    clctParamDef = cms.PSet(
        clctBxWidth     = cms.uint32(6),
        clctDriftDelay  = cms.uint32(2),
        clctHsThresh    = cms.uint32(2),
        clctDsThresh    = cms.uint32(2),
        clctNphPattern  = cms.uint32(4),
        clctFifoTbins   = cms.uint32(12),
        clctFifoPretrig = cms.uint32(7),
        # TMB07 parameters
        clctPidThresh   = cms.uint32(2)
        clctSepSrc      = cms.uint32(1),
        clctSepVme      = cms.uint32(10),
        # Debug
        verbosity = cms.untracked.int32(0)
    ),

    # Parameters for CLCT processors: MTCC-II
    clctParamMTCC2 = cms.PSet(
        clctBxWidth     = cms.uint32(6),
        clctDriftDelay  = cms.uint32(2),
        clctHsThresh    = cms.uint32(4),
        clctDsThresh    = cms.uint32(4),
        clctNphPattern  = cms.uint32(1),
        clctFifoTbins   = cms.uint32(12),
        clctFifoPretrig = cms.uint32(7),
        # TMB07 parameters
        clctHitThresh   = cms.uint32(2),
        clctPidThresh   = cms.uint32(2),
        clctSepSrc      = cms.uint32(1),
        clctSepVme      = cms.uint32(10),
        # Debug
        verbosity = cms.untracked.int32(0)
    ),

    tmbParam = cms.PSet(
        verbosity = cms.untracked.int32(0)
    )
)
