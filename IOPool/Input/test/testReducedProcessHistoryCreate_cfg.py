import FWCore.ParameterSet.Config as cms
import argparse

parser = argparse.ArgumentParser(description='Create files for reduced ProcessHistory test')
parser.add_argument("--version", type=str, help="CMSSW version to be used in the ProcessHistory")
parser.add_argument("--firstEvent", default=1, type=int, help="Number of first event")
parser.add_argument("--lumi", default=1, type=int, help="LuminosityBlock number")
parser.add_argument("--output", type=str, help="Output file name")

args = parser.parse_args()

process = cms.Process("PROD")
process._specialOverrideReleaseVersionOnlyForTesting(args.version)

process.maxEvents.input = 10

from FWCore.Modules.modules import EmptySource
process.source = EmptySource(
    firstEvent = args.firstEvent,
    firstLuminosityBlock = args.lumi,
)

from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(
    fileName = args.output
)

from FWCore.Framework.modules import IntProducer
process.intProducer = IntProducer(ivalue = 42)

from FWCore.Integration.modules import ThingWithMergeProducer
process.thingWithMergeProducer = ThingWithMergeProducer()

process.t = cms.Task(
    process.intProducer,
    process.thingWithMergeProducer,
)
process.p = cms.Path(process.t)
process.ep = cms.EndPath(process.out)
