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

process.f1 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(40),
    onlyOne = cms.untracked.bool(False)
)

process.f2 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(30),
    onlyOne = cms.untracked.bool(False)
)

process.f3 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(20),
    onlyOne = cms.untracked.bool(False)
)

process.f4 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(2),
    onlyOne = cms.untracked.bool(True)
)

process.outp1 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(40),
    name = cms.string('p1'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p1')
    )
)

process.outp2 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(99),
    name = cms.string('p2'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p2')
    )
)

process.outp3 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(10),
    name = cms.string('p3'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p3')
    )
)

process.outp4 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(5),
    name = cms.string('p4'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p4')
    )
)

process.outp5 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(10),
    name = cms.string('p5'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p5')
    )
)

process.outp6 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(99),
    name = cms.string('p6'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p6')
    )
)

process.p1 = cms.Path(process.f1*process.m1)
process.p2 = cms.Path(cms.ignore(process.f1)*process.m1)
process.p3 = cms.Path(process.f2*process.m2*~process.f3*process.m3*cms.ignore(process.f4))
process.p4 = cms.Path(process.f2*process.m2*~process.f3*cms.ignore(process.m3)*process.f4)
process.p5 = cms.Path(process.m4+process.f2*process.m2*~process.f3*process.m3*cms.ignore(process.f4)+process.m4)
process.p6 = cms.Path(process.m1)
process.e = cms.EndPath(process.outp1*process.outp2*process.outp3*process.outp4*process.outp5)
process.e6 = cms.EndPath(process.f1*~process.f1*process.f2*~process.f2*process.f3*~process.f3*process.f4*~process.f4*process.outp6)
# foo bar baz
# vXyxLRFB5vB5G
