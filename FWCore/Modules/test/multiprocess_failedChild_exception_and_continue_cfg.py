# Configuration file for EmptySource

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.options = cms.untracked.PSet(multiProcesses=cms.untracked.PSet(
                                                                       maxChildProcesses=cms.untracked.int32(3),
                                                                       maxSequentialEventsPerChild=cms.untracked.uint32(2),
                                                                       continueAfterChildFailure=cms.untracked.bool(True))
                                     )


process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(100),
    numberEventsInRun = cms.untracked.uint32(5),
    firstTime = cms.untracked.uint64(1000),
    timeBetweenEvents = cms.untracked.uint64(10)
)


process.die = cms.EDAnalyzer("AbortOnEventIDAnalyzer",
                             eventsToAbort = cms.untracked.VEventID([cms.EventID(102,2)]),
                             throwExceptionInsteadOfAbort=cms.untracked.bool(True)
                              )


process.p = cms.EndPath(process.die)
