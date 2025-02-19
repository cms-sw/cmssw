import FWCore.ParameterSet.Config as cms

process = cms.Process("EXTRA")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring('file:testComplex8.root'),
   processingMode = cms.untracked.string("RunsAndLumis")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testComplex8RL.root')
)

process.e = cms.EndPath(process.out)
