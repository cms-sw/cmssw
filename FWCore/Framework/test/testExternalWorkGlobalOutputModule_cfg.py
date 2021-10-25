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

process.f1 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(30),
    onlyOne = cms.untracked.bool(True)
)

process.outp = cms.OutputModule("ExternalWorkSewerModule",
                                shouldPass = cms.int32(3),
    name = cms.string('for_p'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p') 
    )
)

process.outNone = cms.OutputModule("ExternalWorkSewerModule",
    shouldPass = cms.int32(99),
    name = cms.string('for_none')
)

process.outpempty = cms.OutputModule("ExternalWorkSewerModule",
    shouldPass = cms.int32(99),
    name = cms.string('pEmpty'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pEmpty')
    )
)

process.pEmpty = cms.Path()
process.p = cms.Path(process.f1)

process.e1 = cms.EndPath(process.outp)
process.e2 = cms.EndPath(process.outNone)
process.e3 = cms.EndPath(process.outpempty)


