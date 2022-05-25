import FWCore.ParameterSet.Config as cms

import argparse
import sys
parser = argparse.ArgumentParser(prog=sys.argv[0], description="Test PoolOutputModule")
parser.add_argument("--firstLumi", type=int, default=None, help="Set first lumi to process ")

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)


process = cms.Process("TESTOUTPUT")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents.input = 20

process.Thing = cms.EDProducer("ThingProducer")

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:PoolOutputTest.root')
)

process.source = cms.Source("EmptySource")
if args.firstLumi is not None:
    process.source.firstLuminosityBlock = cms.untracked.uint32(args.firstLumi)
    process.output.fileName = "file:PoolOutputTestLumi{}.root".format(args.firstLumi)

process.p = cms.Path(process.Thing*process.OtherThing)
process.ep = cms.EndPath(process.output)


