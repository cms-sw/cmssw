
# This tests that the process history information that should have
# been added by previous process is available. No output is written.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
    )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:testSeriesOfProcessesTEST.root'),
                            )

process.hk = cms.EDAnalyzer("TestHistoryKeeping",
                            expected_processes = cms.vstring('HLT','PROD','TEST'),
                            number_of_expected_HLT_processes_for_each_run = cms.int32(1)
                            )

process.hist = cms.Path(process.hk)
