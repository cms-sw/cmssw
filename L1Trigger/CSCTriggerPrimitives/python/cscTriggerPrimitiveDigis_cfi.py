import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
cscTriggerPrimitiveDigis = cms.EDProducer("CSCTriggerPrimitivesProducer",
    CSCCommonTrigger,
    alctParamDef = cms.PSet(
        alctMode = cms.uint32(1),
        alctTrigMode = cms.uint32(3),
        verbosity = cms.untracked.int32(0),
        alctNphPattern = cms.uint32(4),
        alctBxWidth = cms.uint32(6),
        alctL1aWindow = cms.uint32(5),
        alctDriftDelay = cms.uint32(3),
        alctNphThresh = cms.uint32(2),
        alctFifoTbins = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10)
    ),
    clctParamDef = cms.PSet(
        clctDriftDelay = cms.uint32(2),
        clctFifoPretrig = cms.uint32(7),
        clctFifoTbins = cms.uint32(12),
        clctNphPattern = cms.uint32(4),
        clctBxWidth = cms.uint32(6),
        verbosity = cms.untracked.int32(0),
        clctHsThresh = cms.uint32(2),
        clctSepSrc = cms.uint32(1),
        clctSepVme = cms.uint32(10),
        clctDsThresh = cms.uint32(2),
        clctPidThresh = cms.uint32(2)
    ),
    CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    clctParamMTCC2 = cms.PSet(
        clctDriftDelay = cms.uint32(2),
        clctFifoPretrig = cms.uint32(7),
        clctFifoTbins = cms.uint32(12),
        clctNphPattern = cms.uint32(1),
        clctBxWidth = cms.uint32(6),
        verbosity = cms.untracked.int32(0),
        clctHsThresh = cms.uint32(4),
        clctHitThresh = cms.uint32(2),
        clctSepSrc = cms.uint32(1),
        clctSepVme = cms.uint32(10),
        clctDsThresh = cms.uint32(4),
        clctPidThresh = cms.uint32(2)
    ),
    tmbParam = cms.PSet(
        verbosity = cms.untracked.int32(0)
    ),
    commonParam = cms.PSet(
        isTMB07 = cms.bool(True),
        isMTCC = cms.bool(True)
    ),
    CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    alctParamMTCC2 = cms.PSet(
        alctMode = cms.uint32(0),
        alctTrigMode = cms.uint32(2),
        verbosity = cms.untracked.int32(0),
        alctNphPattern = cms.uint32(4),
        alctBxWidth = cms.uint32(6),
        alctL1aWindow = cms.uint32(3),
        alctDriftDelay = cms.uint32(3),
        alctNphThresh = cms.uint32(2),
        alctFifoTbins = cms.uint32(16),
        alctFifoPretrig = cms.uint32(10)
    ),
    alctParamMTCC1 = cms.PSet(
        alctMode = cms.uint32(2),
        alctTrigMode = cms.uint32(0),
        verbosity = cms.untracked.int32(0),
        alctNphPattern = cms.uint32(4),
        alctBxWidth = cms.uint32(6),
        alctL1aWindow = cms.uint32(3),
        alctDriftDelay = cms.uint32(3),
        alctNphThresh = cms.uint32(4),
        alctFifoTbins = cms.uint32(10),
        alctFifoPretrig = cms.uint32(12)
    )
)


