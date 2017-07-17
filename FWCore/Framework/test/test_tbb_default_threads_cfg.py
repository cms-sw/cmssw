import FWCore.ParameterSet.Config as cms

process = cms.Process("LONGTEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.options = cms.untracked.PSet( numberOfThreads = cms.untracked.uint32(0))

process.add_(cms.Service("TestNThreadsChecker", nExpectedThreads=cms.untracked.uint32(0)))
