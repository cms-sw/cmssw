import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser(description='Test argparse')
parser.add_argument("--maxEvents", help="max events to process", type=int, default=1)
# same as a cmsRun argument
parser.add_argument("-n", "--numThreads", help="number of threads", type=int, default=1)
args = parser.parse_args()

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")

process.maxEvents.input = args.maxEvents
process.options.numberOfThreads = args.numThreads
