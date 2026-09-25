import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  fileMode  = cms.untracked.string('FULLMERGE'),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)


#this should fail
#Contents of files
#testRunMerge.root "PROD" and "MERGE" [run:1, lumi:1, ev:1-25]&[run:2, lumi:1, ev:1-5]&[run:11, lumi:1, ev:1]&[run:11-21, lumi:2-4, ev:1-9] (run 1 is split with run 2 between)

process.source = cms.Source("RNTupleTempSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge.root',
        'file:testRunMerge.root'
    ),
    needSecondaryFileNames = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('checkEachRealDataFile')
)

process.out = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string('file:testRunMergeRecombinedFail.root')
)

process.endpath1 = cms.EndPath(process.out)
