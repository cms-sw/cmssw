import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'


#Contents of files
#testRunMerge1.root  "PROD" [run:1,lumi:1, ev:11-20]
#testRunMerge2.root  "PROD" [run:1, lumi:1, ev:21-25] [run:2,lumi:1, ev:1-5]

from FWIO.RNTupleTempInput.modules import RNTupleTempSource
process.source = RNTupleTempSource(
    fileNames = [
        'file:testRunMerge1.root',
        'file:testRunMerge2.root'
    ]
    #, processingMode = 'RunsAndLumis'
    #, duplicateCheckMode = 'checkEachRealDataFile'
    , noEventSort = False
    , inputCommands = [
        'keep *',
        'drop edmtestThingWithMerge_makeThingToBeDropped1_*_*'
    ]
)

from FWCore.TestModules.modules import RunLumiEventAnalyzer
process.test = RunLumiEventAnalyzer(
    verbose = True
    , expectedRunLumiEvents = [
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
    ]
)

process.looper = cms.Looper("NavigateEventsLooper")

from FWIO.RNTupleTempOutput.modules import RNTupleTempOutputModule
process.out = RNTupleTempOutputModule(
    fileName = 'testLooperEventNavigation.root',
    fastCloning = False
)

process.path1 = cms.Path(process.test)
process.endpath1 = cms.EndPath(process.out)
