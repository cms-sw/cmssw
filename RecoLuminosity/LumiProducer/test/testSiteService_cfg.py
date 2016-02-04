import FWCore.ParameterSet.Config as cms

process = cms.Process("testsiteservice")
process.load("FWCore.MessageService.MessageLogger_cfi")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(False),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
process.maxLuminosityBlocks=cms.untracked.PSet(
    input=cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource",
     numberEventsInRun = cms.untracked.uint32(1),
     firstRun = cms.untracked.uint32(122314),
     numberEventsInLuminosityBlock = cms.untracked.uint32(1),
     firstLuminosityBlock = cms.untracked.uint32(1)
)

process.test = cms.EDAnalyzer("testSiteService")

process.p1 = cms.Path( process.test)

