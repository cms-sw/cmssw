import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge1.root',
        'file:testRunMerge2.root'
    )
    #, processingMode = cms.untracked.string('RunsAndLumis')
    #, duplicateCheckMode = cms.untracked.string('checkEachRealDataFile')
    , noEventSort = cms.untracked.bool(False)
    , inputCommands = cms.untracked.vstring(
        'keep *',
        'drop edmtestThingWithMerge_makeThingToBeDropped1_*_*'
    )
)


process.test = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True)
    , expectedRunLumiEvents = cms.untracked.vuint32(
    1, 0, 0,
    1, 1, 0,
    1, 1, 11,
    1, 1, 12,
    1, 1, 15,
    1, 1, 14,
    1, 1, 15,
    1, 1, 0,
    1, 0, 0,
    2, 0, 0,
    2, 1, 0,
    2, 1, 5,
    2, 1, 3,
    2, 1, 4,
    2, 1, 3,
    2, 1, 0,
    2, 0, 0,
    1, 0, 0,
    1, 1, 0,
    1, 1, 17,
    1, 1, 0,
    1, 0, 0,
    1, 0, 0,
    1, 1, 0,
    1, 1, 11,
    1, 1, 20,
    1, 1, 21,
    1, 1, 0,
    1, 0, 0
    )
)

process.looper = cms.Looper("NavigateEventsLooper")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:testLooperEventNavigation.root'),
    fastCloning = cms.untracked.bool(False)
)

process.path1 = cms.Path(process.test)
process.endpath1 = cms.EndPath(process.out)
