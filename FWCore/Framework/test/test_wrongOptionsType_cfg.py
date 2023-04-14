import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test wrong process.options parameter types')

parser.add_argument("--name", help="Name of parameter", type=str)
parser.add_argument("--value", help="Value of the parameter", type=str)

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2))

# sizeOfStackForThreadsInKB is checked only if numberOfThreads is set
process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.int32(1))
setattr(process.options, args.name, eval(str(args.value)))
