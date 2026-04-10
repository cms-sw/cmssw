import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ModuleEventAllocMonitor service.')
parser.add_argument("--output", default="moduleEventAlloc.log", help="Output log file")
parser.add_argument("--skipEvents", action="store_true", help="Test skipping events")
parser.add_argument("--edmodule", action="store_true", help="Show only specific ed module")
parser.add_argument("--maxEvents", type=int, default=3, help="Specify maxEvents")
parser.add_argument("--threads", type=int, default=1, help="Set number of threads and streams")
args = parser.parse_args()


process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents.input = args.maxEvents
if args.threads > 1:
    process.options.numberOfThreads = args.threads

process.thing = cms.EDProducer("ThingProducer",
    offsetDelta = cms.int32(1)
)

process.OtherThing = cms.EDProducer("OtherThingProducer", thingTag = cms.InputTag("thing"))

process.thingProducer = cms.EDProducer("ThingProducer",
                                       offsetDelta = cms.int32(100),
                                       nThings = cms.int32(50)
)
process.get = cms.EDAnalyzer("edmtest::ThingAnalyzer")

process.Int = cms.EDProducer(
    "IntProducer",
    ivalue = cms.int32(67)
)
process.add_(cms.Service("WaitingService"))
process.acquireInt = cms.EDProducer(
    "AcquireIntStreamProducer",
    tags = cms.VInputTag(),
    produceTag = cms.InputTag("Int")
)
process.getInt = cms.EDAnalyzer(
    "MultipleIntsAnalyzer",
    getFromModules = cms.untracked.VInputTag("acquireInt")
)

process.out = cms.OutputModule("AsciiOutputModule")


process.ep = cms.EndPath(
    process.out +
    process.get +
    process.getInt,
    cms.Task(process.thing,
             process.OtherThing,
             process.thingProducer,
             process.Int,
             process.acquireInt)
)

#process.add_(cms.Service("Tracer"))
process.add_(cms.Service("ModuleEventAllocMonitor", fileName = cms.untracked.string(args.output)))
if args.skipEvents:
    process.ModuleEventAllocMonitor.nEventsToSkip = cms.untracked.uint32(2)

if args.edmodule:
    process.ModuleEventAllocMonitor.moduleNames = cms.untracked.vstring(["thingProducer"])
