import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.load("DQMServices.Core.test.MessageLogger_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.tester = cms.EDFilter("DQMReadFileExample",
    RootFileName = cms.untracked.string('test_playback.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.p = cms.Path(process.tester)


