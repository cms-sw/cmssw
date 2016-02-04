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

process.aux = cms.EDProducer("EventAuxiliaryHistoryProducer",
    historyDepth = cms.uint32(5)
)


process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('test.root')
)


process.p1 = cms.Path(process.aux)

process.e1 = cms.EndPath(process.out)
