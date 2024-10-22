
import FWCore.ParameterSet.Config as cms
import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test InputSource Declaring last run or lumi entry for merge')

parser.add_argument("--enableDeclareLast", action="store_true", help="Declare last entry for merge")
parser.add_argument("--enableDeclareAllLast", action="store_true", help="Declare all entries as last for merge (force intentional source bug)")
parser.add_argument("--multipleEntriesForRun", type=int)
parser.add_argument("--multipleEntriesForLumi", type=int)

args = parser.parse_args()

process = cms.Process("PROD")

process.options = dict(
    numberOfThreads = 2,
    numberOfStreams = 2,
    numberOfConcurrentRuns = 1,
    numberOfConcurrentLuminosityBlocks = 2
)

process.Tracer = cms.Service("Tracer",
    printTimestamps = cms.untracked.bool(True)
)

process.source = cms.Source("SourceWithWaits",
    timePerLumi = cms.untracked.double(1),
    sleepAfterStartOfRun = cms.untracked.double(0.25),
    eventsPerLumi = cms.untracked.vuint32(4,0,5,4,0,5),
    lumisPerRun = cms.untracked.uint32(3),
    declareLast = cms.untracked.bool(False),
    declareAllLast = cms.untracked.bool(False),
    multipleEntriesForLumi = cms.untracked.uint32(0),
    multipleEntriesForRun = cms.untracked.uint32(0)
)

if args.enableDeclareLast:
    process.source.declareLast = True

if args.enableDeclareAllLast:
    process.source.declareAllLast = True

if args.multipleEntriesForLumi is not None:
    process.source.multipleEntriesForLumi = args.multipleEntriesForLumi

if args.multipleEntriesForRun is not None:
    process.source.multipleEntriesForRun = args.multipleEntriesForRun

process.sleepingProducer = cms.EDProducer("timestudy::SleepingProducer",
    ivalue = cms.int32(1),
    consumes = cms.VInputTag(),
    eventTimes = cms.vdouble(0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
)

process.p = cms.Path(process.sleepingProducer)
