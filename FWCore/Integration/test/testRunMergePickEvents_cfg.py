
# Test the PoolSource parameter that one can use
# to request specific events.  This interacts with
# the duplicate checking and skip event feature so
# I set those parameters also and test that.

import FWCore.ParameterSet.Config as cms

process = cms.Process("COPY")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  fileMode  = cms.untracked.string('FULLMERGE'),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

from IOPool.Input.modules import PoolSource
process.source = PoolSource(
    fileNames = [
        'file:testRunMergeRecombined.root',
        'file:testRunMergeRecombined.root'
    ]
    #, duplicateCheckMode = 'checkAllFilesOpened'
    , skipEvents = 3
    , eventsToProcess = ['1:2-1:7']
)

from FWCore.Framework.modules import RunLumiEventAnalyzer
process.test = RunLumiEventAnalyzer(
    verbose = True,
    expectedRunLumiEvents = [
1, 0, 0,
1, 1, 0,
1, 1, 5,
1, 1, 6,
1, 1, 7,
1, 1, 0,
1, 0, 0,
]
)

process.path1 = cms.Path(process.test)

from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(
    fileName = 'testRunMergePickEvent.root'
)

process.endpath1 = cms.EndPath(process.out)
