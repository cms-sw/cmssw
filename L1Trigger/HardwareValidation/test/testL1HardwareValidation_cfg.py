import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.EventContent.EventContent_cff")

# unpacking, conditions needed here
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("L1Trigger.Configuration.L1Config_cff")

process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('l1HardwareVal.root')
)

process.p0 = cms.Path(process.RawToDigi)
process.p1 = cms.Path(process.L1HardwareValidation)
process.e = cms.EndPath(process.out)

process.PoolSource.fileNames = ['file:TTbar_cfi__GEN_SIM_DIGI_L1_DIGI2RAW.root']


