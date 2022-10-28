import FWCore.ParameterSet.Config as cms

import argparse
import sys
parser = argparse.ArgumentParser(prog=sys.argv[0], description="Test PoolOutputModule")
parser.add_argument("--guid", type=str, help="GUID that overrides")
parser.add_argument("--maxSize", type=int, default=None, help="Set maximum file size")
parser.add_argument("--input", type=str, default=[], nargs="*", help="Optional list of input files")

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

process = cms.Process("TESTOUTPUTGUID")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents.input = 20

process.Thing = cms.EDProducer("ThingProducer")

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.source = cms.Source("EmptySource")
if len(args.input) > 0:
    process.maxEvents.input = -1
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring("file:"+x for x in args.input)
    )

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:PoolOutputTestOverrideGUID.root'),
    overrideGUID = cms.untracked.string(args.guid),
)
if args.maxSize is not None:
    process.output.maxSize = cms.untracked.int32(args.maxSize)

process.p = cms.Path(process.Thing*process.OtherThing)
process.ep = cms.EndPath(process.output)


