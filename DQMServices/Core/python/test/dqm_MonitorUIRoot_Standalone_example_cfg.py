import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.load("DQMServices.Core.test.MessageLogger_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.tester = cms.EDFilter("DQMOldReceiverStandaloneExample")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("EmptySource")

process.p = cms.Path(process.tester)


