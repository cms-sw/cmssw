
# Make sure we can read and write the end result of
# all the merge and secondary file input testing

import FWCore.ParameterSet.Config as cms

process = cms.Process("COPY")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
#process.MessageLogger.cerr.threshold = 'ERROR'

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  fileMode  = cms.untracked.string('FULLMERGE'),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMergeRecombined.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:testRunMergeRecombinedCopied.root')
)

process.endpath1 = cms.EndPath(process.out)
