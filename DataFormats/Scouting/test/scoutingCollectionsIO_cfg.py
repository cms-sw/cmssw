import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(
    prog = 'cmsRun '+sys.argv[0]+' --',
    description = 'Configuration file to test I/O of Scouting collections.',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-t', '--nThreads', type = int, help = 'Number of threads',
                    default = 1)

parser.add_argument('-s', '--nStreams', type = int, help = 'Number of EDM streams',
                    default = 0)

parser.add_argument('-i', '--inputFiles', nargs = '+', help = 'List of EDM input files',
                    default = ['/store/mc/Run3Summer22DR/GluGlutoHHto2B2Tau_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/GEN-SIM-RAW/124X_mcRun3_2022_realistic_v12-v2/2550000/bbfb86f3-4073-47e3-967b-059aa6b904ad.root'])

parser.add_argument('-n', '--maxEvents', type = int, help = 'Max number of input events to be processed',
                    default = 10)

parser.add_argument('--skipEvents', type = int, help = 'Number of input events to be skipped',
                    default = 0)

parser.add_argument('-o', '--outputFile', type = str, help = 'Path to output EDM file in ROOT format',
                    default = 'scoutingCollectionsIO_output.root')

parser.add_argument('--wantSummary', action = 'store_true', help = 'Value of process.options.wantSummary',
                    default = False)

argv = sys.argv[:]
if '--' in argv:
    argv.remove('--')

args, unknown = parser.parse_known_args(argv)

# Process
process = cms.Process('TEST')

process.options.numberOfThreads = args.nThreads
process.options.numberOfStreams = args.nStreams
process.options.wantSummary = args.wantSummary

process.maxEvents.input = args.maxEvents

# Source (EDM input)
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(args.inputFiles),
    skipEvents = cms.untracked.uint32(args.skipEvents)
)

# MessageLogger (Service)
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# Output module
process.testOutput = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string( args.outputFile ),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *Scouting*_*_*_*',
    )
)

# EndPath
process.testEndPath = cms.EndPath( process.testOutput )
