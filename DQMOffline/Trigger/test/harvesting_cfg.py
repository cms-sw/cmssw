import FWCore.ParameterSet.Config as cms

## CLI parser
import argparse
import sys

parser = argparse.ArgumentParser(
    prog = 'cmsRun '+sys.argv[0]+' --',
    description = 'Configuration file to run the DQMFileSaver on DQMIO input files.',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-t', '--nThreads', type = int, help = 'Number of threads',
                    default = 4)

parser.add_argument('-s', '--nStreams', type = int, help = 'Number of EDM streams',
                    default = 0)

parser.add_argument('-i', '--inputFiles', nargs = '+', help = 'List of DQMIO input files',
                    default = ['file:testHLTFiltersDQMonitor_DQMIO.root'])

argv = sys.argv[:]
if '--' in argv:
    argv.remove('--')
args, unknown = parser.parse_known_args(argv)

# Process
process = cms.Process('HARVESTING')

process.options.numberOfThreads = args.nThreads
process.options.numberOfStreams = args.nStreams
process.options.numberOfConcurrentLuminosityBlocks = 1

# Source (DQM input)
process.source = cms.Source('DQMRootSource',
  fileNames = cms.untracked.vstring(args.inputFiles)
)

# DQMStore (Service)
process.load('DQMServices.Core.DQMStore_cfi')

# MessageLogger (Service)
process.load('FWCore.MessageLogger.MessageLogger_cfi')

# Output module (file in ROOT format)
from DQMServices.Components.DQMFileSaver_cfi import dqmSaver as _dqmSaver
process.dqmSaver = _dqmSaver.clone(
  workflow = '/DQMOffline/Trigger/'+process.name_()
)

# EndPath
process.endp = cms.EndPath(
  process.dqmSaver
)
