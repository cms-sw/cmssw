

import FWCore.ParameterSet.Config as cms
import argparse
from FWCore.Modules.modules import EmptySource


parser = argparse.ArgumentParser(description="cmsRun config for testing writing product registries with different content")
parser.add_argument("--startEvent", type=int, default=1, help="First event number for EmptySource")
parser.add_argument("--outputFile", type=str, default=None, help="Output file name for PoolOutputModule")
args, unknown = parser.parse_known_args()

process = cms.Process("TEST")

process.maxEvents.input = 3
# Source using imported EmptySource
process.source = EmptySource(
    firstRun = 1,
    firstLuminosityBlock = 1,
    firstEvent = args.startEvent
)


# EventIDFilter: pass only specific events using recommended import style
from FWCore.Modules.modules import EventIDFilter
process.eventIDFilter = EventIDFilter(
    eventsToPass = [
        cms.EventID(1,1,1),
        cms.EventID(1,1,2),
        cms.EventID(1,1,3)
    ]
)


# ThingProducer and OtherThingProducer using recommended import style
from FWCore.Integration.modules import ThingProducer, OtherThingProducer
process.thing = ThingProducer()
process.otherThing = OtherThingProducer(thingTag="thing")

# Path: EventIDFilter -> ThingProducer -> OtherThingProducer
process.p = cms.Path(
    process.eventIDFilter +
    process.thing +
    process.otherThing
)

from IOPool.Output.modules import PoolOutputModule
output_file = args.outputFile if args.outputFile is not None else f'testDifferentProductRegistriesWrite{args.startEvent}.root'
process.out = PoolOutputModule(
    outputCommands = [
        "keep *",
        'drop *_thing_*_*'
    ],
    fileName = cms.untracked.string(output_file)
)

process.e = cms.EndPath(process.out)