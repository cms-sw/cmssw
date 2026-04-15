import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ModuleAllocProfiler service.')
parser.add_argument("--output", default="", help="Output log file pattern")
parser.add_argument("--skipEvents", action="store_true", help="test skipping events")
parser.add_argument("--source", action="store_true", help="show source")
parser.add_argument("--out", action="store_true", help="show OutputModule")
parser.add_argument("--edmodule", action="store_true", help="show a specific ed module")
parser.add_argument("--esmodule", action="store_true", help="show a specific es module")
args = parser.parse_args()


process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 3

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.Thing = cms.EDProducer("ThingProducer",
    offsetDelta = cms.int32(1)
)

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.thingProducer = cms.EDProducer("ThingProducer",
                                       offsetDelta = cms.int32(100),
                                       nThings = cms.int32(50)
)

process.add_(cms.Service("timestudy::SleepingServer",
    nWaitingEvents = cms.untracked.uint32(1)
))
process.externalWorkProducer = cms.EDProducer("timestudy::ExternalWorkSleepingProducer",
    consumes = cms.VInputTag(),
    ivalue = cms.int32(10),
    eventTimes = cms.vdouble(0.01),
    serviceInitTimes = cms.vdouble(0.,0.),
    serviceWorkTimes = cms.vdouble(0.1,0.05),
    serviceFinishTimes = cms.vdouble(0.,0.)
)
process.externalWorkAnalyzer = cms.EDAnalyzer("timestudy::OneSleepingAnalyzer",
    consumes = cms.VInputTag("externalWorkProducer"),
    eventTimes = cms.vdouble(0.0001)
)


process.get = cms.EDAnalyzer("WhatsItAnalyzer")

process.emptyESSourceI = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordI"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)
process.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)
process.acquireIntESProducer = cms.ESProducer("AcquireIntESProducer",
    numberOfIOVsToAccumulate = cms.untracked.uint32(2),
    secondsToWaitForWork = cms.untracked.uint32(1)
)
process.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(1),
    expectedValues = cms.untracked.vint32(11, 11, 11, 11, 11, 11, 11, 11)
)


process.out = cms.OutputModule("AsciiOutputModule")

process.ep = cms.EndPath(
    process.out+
    process.get+
    process.externalWorkAnalyzer+
    process.esTestAnalyzerB,
    cms.Task(
        process.WhatsItESProducer,
        process.DoodadESSource,
        process.emptyESSourceI,
        process.emptyESSourceB,
        process.acquireIntESProducer,
        process.Thing,
        process.OtherThing,
        process.thingProducer,
        process.externalWorkProducer
    )
)

process.add_(cms.Service("ModuleAllocProfiler",
    moduleNames = cms.untracked.vstring(),
    filePattern = cms.untracked.string(args.output),
))

if args.skipEvents:
    process.ModuleAllocProfiler.nEventsToSkip = cms.untracked.uint32(2)

if args.source:
    process.ModuleAllocProfiler.moduleNames.append("source")
if args.edmodule:
    process.ModuleAllocProfiler.moduleNames.extend(["thingProducer", "externalWorkProducer"])
if args.esmodule:
    process.ModuleAllocProfiler.moduleNames.extend(["WhatsItESProducer", "acquireIntESProducer"])
if args.out:
    process.ModuleAllocProfiler.moduleNames.append("out")
