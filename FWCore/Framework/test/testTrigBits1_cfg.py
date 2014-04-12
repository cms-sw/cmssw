#Simply two trigger paths defined
#without any filters

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

process.m1a = cms.EDProducer("IntProducer",
    ivalue = cms.int32(1)
)

process.m2a = cms.EDProducer("IntProducer",
    ivalue = cms.int32(1)
)

process.a1 = cms.EDAnalyzer("TestResultAnalyzer",
    name = cms.untracked.string('a1'),
    dump = cms.untracked.bool(True),
    numbits = cms.untracked.int32(2)
)

process.testout1 = cms.OutputModule("TestOutputModule",
    bitMask = cms.int32(5),
    name = cms.string('testout1'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p2a')
    )
)

process.p1a = cms.Path(process.m1a)
process.p2a = cms.Path(process.m2a)
process.e1 = cms.EndPath(process.a1)
process.e2 = cms.EndPath(process.testout1)
