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

process.m1b = cms.EDProducer("IntProducer",
    ivalue = cms.int32(10)
)

process.m2b = cms.EDProducer("IntProducer",
    ivalue = cms.int32(20)
)

process.m3b = cms.EDProducer("IntProducer",
    ivalue = cms.int32(30)
)

process.m4b = cms.EDProducer("IntProducer",
    ivalue = cms.int32(40)
)

process.m5b = cms.EDProducer("IntProducer",
    ivalue = cms.int32(50)
)

process.m6b = cms.EDProducer("IntProducer",
    ivalue = cms.int32(60)
)

process.a1 = cms.EDAnalyzer("TestResultAnalyzer",
    name = cms.untracked.string('a1'),
    dump = cms.untracked.bool(True)
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

process.outp1a = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(3),
    name = cms.string('for_f1'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p1a')
    )
)

process.outp2a = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(1),
    name = cms.string('for_f2'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p2a')
    )
)

process.outp3a = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(70),
    name = cms.string('for_f4_f5'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p4a', 
            'p5a')
    )
)

process.outp8a = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(29),
    name = cms.string('for_!f5'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('!p5a')
    )
)

process.outp1b = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(96),
    name = cms.string('for_!f1'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p1b')
    )
)

process.outp2b = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(98),
    name = cms.string('for_!f2'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p2b')
    )
)

process.outp3b = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(69),
    name = cms.string('for_!f4_!f5'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p4b', 
            'p5b')
    )
)

process.outp8b = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(70),
    name = cms.string('for_!!f5'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('!p5b')
    )
)

process.outp4 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(99),
    name = cms.string('for_*'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('*')
    )
)

process.outp5 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(0),
    name = cms.string('for_!*'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('!*')
    )
)

process.outp6 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(99),
    name = cms.string('for_*_!*'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('*', 
            '!*')
    )
)

process.outp7 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(99),
    name = cms.string('for_none')
)

process.p1b = cms.Path(~process.f1*process.m1b)
process.p2b = cms.Path(~process.f2*process.m2b)
process.p3b = cms.Path(~process.f3*process.m3b)
process.p4b = cms.Path(~process.f4*process.m4b)
process.p5b = cms.Path(~process.f5*process.m5b)
process.p6b = cms.Path(~process.f6*process.m6b)
process.e1 = cms.EndPath(process.a1)
process.e3 = cms.EndPath(process.outp1b*process.outp2b*process.outp3b*process.outp8b)
process.e4 = cms.EndPath(process.outp4*process.outp5)
process.e5 = cms.EndPath(process.outp6)
process.e6 = cms.EndPath(process.outp7)
# foo bar baz
# G8dFsLFgNX2QI
