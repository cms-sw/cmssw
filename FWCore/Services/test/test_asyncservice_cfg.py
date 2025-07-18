import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test AsyncService')
parser.add_argument("--earlyTermination", help="Test behavior of EarlyTermination signal on subsequent AsyncService::run() calls", action="store_true")
parser.add_argument("--exception", help="Another module throws an exception while asynchronous function is running", action="store_true")
args = parser.parse_args()

process = cms.Process("TEST")

process.maxEvents.input = 8
process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
process.source = cms.Source("EmptySource")

if args.earlyTermination or args.exception:
    process.tester = cms.EDProducer("edmtest::AsyncServiceWaitingTester",
        throwingStream = cms.untracked.uint32(0)
    )

    # Make stream 0 always throw the exception in FailingProducer
    process.streamFilter = cms.EDFilter("edmtest::StreamIDFilter",
        rejectStreams = cms.vuint32(1,2,3)
    )
    process.fail = cms.EDProducer("FailingProducer")
    process.p2 = cms.Path(process.streamFilter+process.fail)

    testerService = cms.Service("edmtest::AsyncServiceTesterService")
    if args.earlyTermination:
        process.tester.waitEarlyTermination = cms.untracked.bool(True)
        testerService.watchEarlyTermination = cms.bool(True)
    elif args.exception:
        process.tester.waitStreamEndRun = cms.untracked.bool(True)
        testerService.watchStreamEndRun = cms.bool(True)
    process.add_(testerService)
else:
    process.tester = cms.EDProducer("edmtest::AsyncServiceTester")

process.p = cms.Path(process.tester)

process.add_(cms.Service("ZombieKillerService", secondsBetweenChecks=cms.untracked.uint32(5)))
