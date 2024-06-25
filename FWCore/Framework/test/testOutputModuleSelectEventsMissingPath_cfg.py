import FWCore.ParameterSet.Config as cms

import argparse
import sys
parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test OutputModule SelectEvents referring to a non-existent Path and/or Process ')
parser.add_argument("--missingPath", type=str, required=True, help="Specify how the specified Path is missing. Can be 'sameProcess', 'earlierProcess', 'missingProcess'")
parser.add_argument("--anotherModule", type=str, default="", help="Specify placement of another producer. Can be empty (default), 'before', 'after")
args = parser.parse_args()

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
if args.missingPath == "earlierProcess":
    process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring("file:testOutputModuleSelectEventsMissingPath.root"))
process.maxEvents.input = 3

selectEvents = {
    "sameProcess" : "nonexistent_path:TEST",
    "earlierProcess" : "nonexistent_path:EARLIER",
    "missingProcess" : "nonexistent_path:NONEXISTENT_PROCESS",
}

process.out = cms.OutputModule("SewerModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(selectEvents[args.missingPath])
    ),
    name = cms.string("out"),
    shouldPass = cms.int32(1)
)

process.ep = cms.EndPath(process.out)

process.intprod = cms.EDProducer("IntProducer", ivalue=cms.int32(3))
if args.anotherModule == "before":
    process.ep.insert(0, process.intprod)
elif args.anotherModule == "after":
    process.ep += process.intprod
elif args.anotherModule != "":
    raise Exception("Invalid value for anotherModule '{}'".format(args.anotherModule))

process.add_(cms.Service('ZombieKillerService', secondsBetweenChecks=cms.untracked.uint32(2)))
