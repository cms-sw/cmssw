import FWCore.ParameterSet.Config as cms

process = cms.Process("EXTRA")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring('file:testComplex7.root'),
   processingMode = cms.untracked.string("Runs")
)

process.i = cms.EDProducer("IntProducer", ivalue = cms.int32(4))

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testComplex7R.root')
)

process.p1 = cms.Path(process.i)

process.e = cms.EndPath(process.out)
