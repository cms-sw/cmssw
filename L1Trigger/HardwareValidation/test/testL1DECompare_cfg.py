import FWCore.ParameterSet.Config as cms

process = cms.Process("testL1DECompare")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("L1Trigger.HardwareValidation.L1Comparator_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:digis.root')
)

process.p = cms.Path(process.l1compare)
process.l1compare.DumpMode = -1
process.l1compare.COMPARE_COLLS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]


