import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource")

process.pre1 = cms.EDFilter("Prescaler",
    prescaleFactor = cms.int32(5)
)

process.pre2 = cms.EDFilter("Prescaler",
    prescaleFactor = cms.int32(2)
)

process.print1 = cms.OutputModule("AsciiOutputModule")

process.print2 = cms.OutputModule("AsciiOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p2')
    )
)

process.p1 = cms.Path(process.pre1)
process.p2 = cms.Path(process.pre2)

process.e1 = cms.EndPath(process.print1)
process.e2 = cms.EndPath(process.print2)
