import FWCore.ParameterSet.Config as cms

## CLI parser
import argparse
import sys

parser = argparse.ArgumentParser(
    prog = 'cmsRun '+sys.argv[0]+' --',
    description = 'Configuration file to test of the HLTFiltersDQMonitor plugin.',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-t', '--nThreads', type = int, help = 'Number of threads',
                    default = 4)

parser.add_argument('-s', '--nStreams', type = int, help = 'Number of EDM streams',
                    default = 0)

parser.add_argument('-i', '--inputFiles', nargs = '+', help = 'List of EDM input files',
                    default = ['/store/relval/CMSSW_12_6_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/125X_mcRun3_2022_realistic_v3-v1/2580000/2d96539c-b321-401f-b7b2-51884a5d421f.root'])

parser.add_argument('-n', '--maxEvents', type = int, help = 'Number of input events',
                    default = 100)

parser.add_argument('-o', '--outputFile', type = str, help = 'Path to output file in DQMIO format',
                    default = 'testHLTFiltersDQMonitor_DQMIO.root')

parser.add_argument('--wantSummary', action = 'store_true', help = 'Value of process.options.wantSummary',
                    default = False)

parser.add_argument('-d', '--debugMode', action = 'store_true', help = 'Enable debug info (requires recompiling first with \'USER_CXXFLAGS="-DEDM_ML_DEBUG" scram b\')',
                    default = False)

argv = sys.argv[:]
if '--' in argv:
    argv.remove('--')
args, unknown = parser.parse_known_args(argv)

## Process
process = cms.Process('TEST')

process.options.numberOfThreads = args.nThreads
process.options.numberOfStreams = args.nStreams
process.options.wantSummary = args.wantSummary
process.maxEvents.input = args.maxEvents

## Source
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(args.inputFiles),
    inputCommands = cms.untracked.vstring(
        'drop *',
        'keep edmTriggerResults_*_*_*',
        'keep triggerTriggerEvent_*_*_*',
        'keep triggerTriggerEventWithRefs_*_*_*'
    )
)

## MessageLogger (Service)
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1 # only report every Nth event start
process.MessageLogger.cerr.FwkReport.limit = -1      # max number of reported messages (all if -1)
process.MessageLogger.cerr.enableStatistics = False  # enable "MessageLogger Summary" message

## DQMStore (Service)
process.load('DQMServices.Core.DQMStore_cfi')

## FastTimerService (Service)
from HLTrigger.Timer.FastTimerService_cfi import FastTimerService as _FastTimerService
process.FastTimerService = _FastTimerService.clone(
    enableDQM = False,
    printEventSummary = False,
    printJobSummary = True,
    printRunSummary = False,
    writeJSONSummary = False
)
process.MessageLogger.FastReport = dict()

## EventData Modules
from DQMOffline.Trigger.dqmHLTFiltersDQMonitor_cfi import dqmHLTFiltersDQMonitor as _dqmHLTFiltersDQMonitor
process.dqmHLTFiltersDQMonitor = _dqmHLTFiltersDQMonitor.clone(
    folderName = 'HLT/Filters',
    efficPlotNamePrefix = 'effic_',
    triggerResults = 'TriggerResults::HLT',
    triggerEvent = 'hltTriggerSummaryAOD::HLT',
    triggerEventWithRefs = 'hltTriggerSummaryRAW::HLT'
)
process.MessageLogger.HLTFiltersDQMonitor = dict()
if args.debugMode:
    process.MessageLogger.cerr.threshold = 'DEBUG'
    process.MessageLogger.debugModules = ['dqmHLTFiltersDQMonitor']

## Output Modules
process.dqmOutput = cms.OutputModule('DQMRootOutputModule',
    fileName = cms.untracked.string(args.outputFile)
)

## Path
process.testPath = cms.Path(
    process.dqmHLTFiltersDQMonitor
)

## EndPath
process.testEndPath = cms.EndPath(
    process.dqmOutput
)
