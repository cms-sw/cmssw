import FWCore.ParameterSet.Config as cms

process = cms.Process("ScalesTest")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")

process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")

process.load("L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.l1muscalestest = cms.EDAnalyzer("L1MuScalesTester")

process.p = cms.Path(process.l1muscalestest)
process.L1MuTriggerScalesRcdSource.iovIsRunNotTime = False
process.L1MuTriggerPtScaleRcdSource.iovIsRunNotTime = False
process.L1MuGMTScalesRcdSource.iovIsRunNotTime = False


