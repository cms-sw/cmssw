import FWCore.ParameterSet.Config as cms

process = cms.Process("LONGTEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.tester = cms.EDAnalyzer("TestTBBTasksAnalyzer",
                                numTasksToRun = cms.untracked.uint32(10),
                                nExpectedThreads = cms.untracked.uint32(8),
                                usecondsToSleep=cms.untracked.uint32(100000))
process.p = cms.Path(process.tester)
