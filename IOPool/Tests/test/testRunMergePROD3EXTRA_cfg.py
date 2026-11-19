import FWCore.ParameterSet.Config as cms

process = cms.Process("EXTRA")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  fileMode = cms.untracked.string('FULLMERGE'),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

#Contents of file
#testRunMerge3.root  "PROD" [run:1, lumi:1, ev:1-10]

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:testRunMerge3.root')
)

process.thingWithMergeProducer3extra = cms.EDProducer("ThingWithMergeProducer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMerge3extra.root')
)
#Contents of file
#testRunMerge3extra.root  "PROD" and "EXTRA" [run:1, lumi:1, ev:1-10]


process.path1 = cms.Path(process.thingWithMergeProducer3extra)

process.e = cms.EndPath(process.out)
