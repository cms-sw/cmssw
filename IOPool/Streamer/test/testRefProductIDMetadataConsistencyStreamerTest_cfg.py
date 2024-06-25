import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ModuleTypeResolver and Ref')
parser.add_argument("--input", action="append", default=[], help="Input files")
args = parser.parse_args()
if len(args.input) == 0:
    parser.error("No input files")

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(["file:"+f for f in args.input])
)

process.tester = cms.EDAnalyzer("OtherThingAnalyzer",
    other = cms.untracked.InputTag("otherThing","testUserTag")
)

process.e = cms.EndPath(process.tester)
