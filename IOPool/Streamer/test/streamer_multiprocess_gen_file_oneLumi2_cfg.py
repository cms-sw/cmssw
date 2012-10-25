# Configuration file for EmptySource

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(102),
    numberEventsInRun = cms.untracked.uint32(20),
    firstTime = cms.untracked.uint64(1000),
    timeBetweenEvents = cms.untracked.uint64(10)
)

process.out = cms.OutputModule("EventStreamFileWriter", fileName = cms.untracked.string("multiprocess_oneLumi2_test.dat"))

process.p = cms.EndPath(process.out)
