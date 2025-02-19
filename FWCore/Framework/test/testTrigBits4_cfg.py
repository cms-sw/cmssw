# The following comments couldn't be translated into the new config version:

#Even number of Trigger paths 4, should just be packed in ONE byte

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

process.m1 = cms.EDProducer("IntProducer",
    ivalue = cms.int32(1)
)

process.m2 = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2)
)

process.m3 = cms.EDProducer("IntProducer",
    ivalue = cms.int32(3)
)

process.m4 = cms.EDProducer("IntProducer",
    ivalue = cms.int32(4)
)

process.a1 = cms.EDAnalyzer("TestResultAnalyzer",
    name = cms.untracked.string('a1'),
    dump = cms.untracked.bool(True),
    numbits = cms.untracked.int32(4)
)

process.testout1 = cms.OutputModule("TestOutputModule",
    bitMask = cms.int32(85),
    name = cms.string('testout1')
)

process.p1 = cms.Path(process.m1)
process.p2 = cms.Path(process.m2)
process.p3 = cms.Path(process.m3)
process.p4 = cms.Path(process.m4)
process.e1 = cms.EndPath(process.testout1)
process.e2 = cms.EndPath(process.a1)
