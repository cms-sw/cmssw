import FWCore.ParameterSet.Config as cms

import argparse
import sys
parser = argparse.ArgumentParser(prog=sys.argv[0], description="Test PoolOutputModule")
parser.add_argument("--guid", type=str, help="GUID that overrides")

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
    fileName = cms.untracked.string('file:PoolOutputTestOverrideGUID.root'),
    overrideGUID = cms.untracked.string(args.guid),
)

process.source = cms.Source("EmptySource")

process.p = cms.Path(process.Thing*process.OtherThing)
process.ep = cms.EndPath(process.output)


