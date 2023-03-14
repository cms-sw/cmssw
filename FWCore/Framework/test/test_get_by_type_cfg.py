import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test getting many DataProducts just by type.')
parser.add_argument("--useConsumesMany", action="store_true", help="use consumesMany instead of GetterOfProducts")
parser.add_argument("--useEDAlias", action="store_true", help="add an EDAlias for one of the modules")

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)


process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 1

process.a = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.b = cms.EDProducer("IntProducer", ivalue = cms.int32(10))
process.c = cms.EDProducer("IntProducer", ivalue = cms.int32(100))

if args.useEDAlias:
    process.d = cms.EDAlias(a = cms.VPSet(cms.PSet(type = cms.string('*'))))
    print("turned on useEDAlias")

useConsumesMany = False
if args.useConsumesMany:
    useConsumesMany = True
    print("turned on useConsumesMany")
process.add = cms.EDProducer("AddAllIntsProducer", useConsumesMany = cms.untracked.bool(useConsumesMany))

process.test = cms.EDAnalyzer("BuiltinIntTestAnalyzer",
                              valueMustMatch = cms.untracked.int32(111),
                              moduleLabel = cms.untracked.InputTag("add")
                              )

process.p = cms.Path(process.add, cms.Task(process.a, process.b, process.c))
process.e = cms.EndPath(process.test)
