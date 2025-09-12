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

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:testRunMerge2.root')
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")
process.dependsOnThingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('thingWithMergeProducer')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMerge2extra.root')
)

process.path1 = cms.Path(process.thingWithMergeProducer *
                         process.dependsOnThingWithMergeProducer)

process.e = cms.EndPath(process.out)
