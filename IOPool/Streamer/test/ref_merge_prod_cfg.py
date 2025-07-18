import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ConditionalTasks.')

parser.add_argument("--extraProducers", help="Add extra producers to configuration", action="store_true")
parser.add_argument("--fileName", help="name of output file")
parser.add_argument("--firstLumi", help="LuminosityBlock number for first lumi", type = int, default=1)
parser.add_argument("--firstRun", help="LuminosityBlock number for first run", type = int, default=1)
parser.add_argument("--keepAllProducts", help="Keep all products made in the job", action="store_true")
parser.add_argument("--dropThings", help="drop the Things collections so the refs will not function", action="store_true")

args = parser.parse_args()


process = cms.Process("PROD")

nEvents = 10
from FWCore.Modules.modules import EmptySource
process.source = EmptySource(firstRun = args.firstRun,
                                firstLuminosityBlock = args.firstLumi,
                                firstEvent = nEvents*(args.firstLumi-1)+1
)

process.maxEvents.input = nEvents

if args.extraProducers:
    from FWCore.Framework.modules import IntProducer
    process.a = IntProducer(ivalue = 1)

    process.b = IntProducer(ivalue = 2)

from FWCore.Integration.modules import ThingProducer, OtherThingProducer, OtherThingAnalyzer
process.c = ThingProducer()

process.d = OtherThingProducer(thingTag="c")

outputs = []
if not args.keepAllProducts:
    outputs = ["drop *",
                "keep edmtestOtherThings_*_*_*"]
    if not args.dropThings:
        outputs.append("keep edmtestThings_*_*_*")
    

from IOPool.Streamer.modules import EventStreamFileWriter
process.o = EventStreamFileWriter(outputCommands = outputs,
                             fileName = args.fileName
                             )
if args.extraProducers:
    process.p = cms.Path(process.a+process.b+process.c*process.d)
else:
    process.p = cms.Path(process.c*process.d)

process.tester = OtherThingAnalyzer(other = ("d","testUserTag"))

process.out = cms.EndPath(process.o+process.tester)
