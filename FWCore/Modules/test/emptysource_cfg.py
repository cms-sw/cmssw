# Configuration file for EmptySource

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(100),
    numberEventsInRun = cms.untracked.uint32(5),
    firstTime = cms.untracked.uint32(1000),
    timeBetweenEvents = cms.untracked.uint32(10)
)

process.print1 = cms.OutputModule("AsciiOutputModule")

process.p = cms.EndPath(process.print1)
