import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(21),
    numberEventsInRun = cms.untracked.uint32(5),
    numberEventsInLuminosityBlock = cms.untracked.uint32(5)
)

process.i = cms.EDProducer("IntProducer", ivalue = cms.int32(4))

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testComplex2.root')
)

process.p1 = cms.Path(process.i)

process.e = cms.EndPath(process.out)
# foo bar baz
# EMchM1bo1WVCO
