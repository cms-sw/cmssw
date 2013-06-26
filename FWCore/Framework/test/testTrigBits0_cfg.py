#No Trig paths defined

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource")

process.a1 = cms.EDAnalyzer("TestResultAnalyzer",
    name = cms.untracked.string('a1'),
    dump = cms.untracked.bool(True),
    numbits = cms.untracked.int32(0),
    pathname = cms.untracked.string('e1'),
    modlabel = cms.untracked.string('a1')
)

process.testout1 = cms.OutputModule("TestOutputModule",
    expectTriggerResults = cms.untracked.bool(False),
    bitMask = cms.int32(0),
    name = cms.string('testout1')
)

process.e1 = cms.EndPath(process.a1)
process.e2 = cms.EndPath(process.testout1)
