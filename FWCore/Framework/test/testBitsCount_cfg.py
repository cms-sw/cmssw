import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(99)
)

process.source = cms.Source("EmptySource")

process.m1a = cms.EDProducer("IntProducer",
    ivalue = cms.int32(1)
)

process.m2a = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2)
)

process.m3a = cms.EDProducer("IntProducer",
    ivalue = cms.int32(3)
)

process.m4a = cms.EDProducer("IntProducer",
    ivalue = cms.int32(4)
)

process.m5a = cms.EDProducer("IntProducer",
    ivalue = cms.int32(5)
)

process.m6a = cms.EDProducer("IntProducer",
    ivalue = cms.int32(6)
)

process.a1 = cms.EDAnalyzer("TestResultAnalyzer",
    name = cms.untracked.string('a1'),
    dump = cms.untracked.bool(True),
    numbits = cms.untracked.int32(6)
)

process.f1 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(30),
    onlyOne = cms.untracked.bool(True)
)

process.f2 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(70),
    onlyOne = cms.untracked.bool(True)
)

process.f3 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(12),
    onlyOne = cms.untracked.bool(True)
)

process.f4 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(30),
    onlyOne = cms.untracked.bool(False)
)

process.f5 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(70),
    onlyOne = cms.untracked.bool(False)
)

process.f6 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(12),
    onlyOne = cms.untracked.bool(False)
)

process.outp4 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(4),
    name = cms.string('for_p1ap2a'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p1a', 
            'p2a')
    )
)

process.outp7 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(99),
    name = cms.string('for_none')
)

process.p1a = cms.Path(process.f1*process.m1a)
process.p2a = cms.Path(process.f2*process.m2a)
process.p3a = cms.Path(process.f3*process.m3a)
process.p4a = cms.Path(process.f4*process.m4a)
process.p5a = cms.Path(process.f5*process.m5a)
process.p6a = cms.Path(process.f6*process.m6a)
process.e1 = cms.EndPath(process.a1)
process.e2 = cms.EndPath(process.outp4)
process.e3 = cms.EndPath(process.outp7)


