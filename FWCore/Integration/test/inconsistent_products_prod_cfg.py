import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test Processes with inconsistent data products.')

parser.add_argument("--dropThing2", help="drop Thing2 from output", action="store_true")
parser.add_argument("--fileName", help="name of output file")
parser.add_argument("--noThing2Prod", help="do not include Thing2's producer", action="store_true")
parser.add_argument("--startEvent", help="starting event number", type=int, default=1)
parser.add_argument("--addAndStoreOther2", help='add the OtherThingProducer consuming thing2 and store it', action='store_true')
parser.add_argument("--thing2DependsOnThing1", help="have thing2 depend on thing1", action='store_true')

args = parser.parse_args()


process = cms.Process("PROD")

nEvents = 10
from FWCore.Modules.modules import EmptySource
process.source = EmptySource(firstEvent = args.startEvent)

process.maxEvents.input = nEvents

from FWCore.Framework.modules import IntProducer, AddIntsProducer
process.thing1 = IntProducer(ivalue=1)

process.t = cms.Task(process.thing1)
if not args.noThing2Prod:
    if args.thing2DependsOnThing1:
        process.thing2 = AddIntsProducer(labels=['thing1'])
    else:
        process.thing2 = IntProducer(ivalue=2)
    process.t.add(process.thing2)
    if args.addAndStoreOther2:
        process.other2 = AddIntsProducer(labels=['thing2'])
        process.t.add(process.other2)
elif args.addAndStoreOther2:
    process.other2 = AddIntsProducer(labels=['thing1'])
    process.t.add(process.other2)

outputs = []
if args.dropThing2:
    outputs = ["keep *",
                "drop *_thing2_*_*"]
    

from IOPool.Output.modules import PoolOutputModule
process.o = PoolOutputModule(outputCommands = outputs,
                             fileName = args.fileName
                             )

process.out = cms.EndPath(process.o, process.t)


                             

                            
