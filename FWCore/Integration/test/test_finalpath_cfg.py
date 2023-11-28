import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test FinalPath.')

parser.add_argument("--schedule", help="use cms.Schedule", action="store_true")
parser.add_argument("--task", help="put EDProducer into a task", action="store_true")
parser.add_argument("--path", help="put a consumer of the product onto a Path", action="store_true")
parser.add_argument("--endpath", help="put a consumer of the product onto an EndPath", action="store_true")
parser.add_argument("--filter", action="store_true")
parser.add_argument("--tracer", help="add Tracer service", action="store_true")

print(sys.argv)
args = parser.parse_args()

process = cms.Process("TEST")

process.MessageLogger.cerr.INFO.limit = 10000

process.source = cms.Source("EmptySource")

process.maxEvents.input = 3

process.thing = cms.EDProducer("ThingProducer")

scheduledPaths =[]
if args.path:
    print("adding Path")
    process.otherThing = cms.EDProducer("OtherThingProducer", thingTag = cms.InputTag("thing"))
    p = cms.Path()
    if args.filter:
        process.fltr = cms.EDFilter("Prescaler", prescaleFactor = cms.int32(2), prescaleOffset=cms.int32(0))
        p += process.fltr
    if not args.task:
        p += process.thing
    p += process.otherThing
    process.p = p
    scheduledPaths.append(process.p)
    if args.task:
        process.p.associate(cms.Task(process.thing))

if args.endpath:
    print("adding EndPath")
    process.out2 = cms.OutputModule("AsciiOutputModule",outputCommands = cms.untracked.vstring("drop *", "keep *_thing_*_*"))
    process.o = cms.EndPath(process.out2)
    scheduledPaths.append(process.o)
    if args.task:
        process.o.associate(cms.Task(process.thing))

process.out = cms.OutputModule("GetProductCheckerOutputModule", verbose= cms.untracked.bool(True), outputCommands = cms.untracked.vstring("drop *", "keep *_thing_*_*"))
process.f = cms.FinalPath(process.out)

if args.schedule:
    print("adding Schedule")
    scheduledPaths.append(process.f)
    process.schedule = cms.Schedule(*scheduledPaths)
    if args.task:
        process.schedule.associate(cms.Task(process.thing))

if args.tracer:
    process.add_(cms.Service("Tracer"))

process.options.numberOfThreads=3
process.options.numberOfStreams=1
