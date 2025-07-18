import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ModuleAllocMonitor service.')
parser.add_argument("--skipEvents", action="store_true", help="test skipping events")
parser.add_argument("--edmodule", action="store_true", help="show only specific ed module")
parser.add_argument("--esmodule", action="store_true", help="show only specific es module")
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

process.get = cms.EDAnalyzer("WhatsItAnalyzer")
process.out = cms.OutputModule("AsciiOutputModule")


process.ep = cms.EndPath(process.out+process.get, cms.Task(process.WhatsItESProducer, process.DoodadESSource, process.Thing, process.OtherThing, process.thingProducer))

#process.add_(cms.Service("Tracer"))
process.add_(cms.Service("ModuleAllocMonitor", fileName = cms.untracked.string("moduleAlloc.log")))

if args.skipEvents:
    process.ModuleAllocMonitor.nEventsToSkip = cms.untracked.uint32(2)

if args.edmodule:
    process.ModuleAllocMonitor.moduleNames = cms.untracked.vstring(["thingProducer"])

if args.esmodule:
    process.ModuleAllocMonitor.moduleNames = cms.untracked.vstring(["WhatsItESProducer"])
