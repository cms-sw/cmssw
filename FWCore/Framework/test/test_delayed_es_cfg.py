import FWCore.ParameterSet.Config as cms
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-d",
    "--delay-target",
    choices=["runs", "lumis"],
    default="lumis",
    help="Select whether DelayESCallsService delays run or lumi transitions",
)
args, _ = parser.parse_known_args()

delay_runs = args.delay_target == "runs"

process = cms.Process("TEST")

process.options.numberOfConcurrentRuns = 3
process.options.numberOfConcurrentLuminosityBlocks = 3
process.options.numberOfThreads = 4

process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(1 if delay_runs else 5),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),
                            processingMode = cms.untracked.string('Runs' if delay_runs else 'RunsAndLumis'))

process.add_(cms.Service("Tracer", fileName = cms.untracked.string('delay_es.log'), useMessageLogger=cms.untracked.bool(False)))

process.add_(cms.Service("edmtest::DelayESCallsService",
                         delay = cms.untracked.uint32(10000),
                         run = cms.untracked.uint32(3 if delay_runs else 1),
                         lumi = cms.untracked.uint32(0 if delay_runs else 3)))

process.maxEvents.input = 5
process.maxLuminosityBlocks.input = 5
