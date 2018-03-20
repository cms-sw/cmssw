import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.hcalPulseContainmentTest = cms.EDAnalyzer("HcalPulseContainmentTest")

process.p1 = cms.Path(process.hcalPulseContainmentTest)
