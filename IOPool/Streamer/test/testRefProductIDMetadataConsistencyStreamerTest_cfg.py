import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ModuleTypeResolver and Ref')
parser.add_argument("--input", action="append", default=[], help="Input files")
parser.add_argument("--numDecodeTasks", type=int, default=1, help="NewEventStreamFileReader numDecodeTasks")
parser.add_argument("--threads", type=int, default=1, help="Number of threads and streams")
parser.add_argument("--skipEvents", type=int, default=0, help="Number of events to skip")
args = parser.parse_args()
if len(args.input) == 0:
    parser.error("No input files")

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(["file:"+f for f in args.input]),
    numDecodeTasks = cms.untracked.uint32(args.numDecodeTasks),
    skipEvents = cms.untracked.uint32(args.skipEvents)
)
process.options.numberOfThreads = args.threads
process.options.numberOfStreams = args.threads
process.options.wantSummary = True
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.tester = cms.EDAnalyzer("OtherThingAnalyzer",
    other = cms.untracked.InputTag("otherThing","testUserTag")
)

process.e = cms.EndPath(process.tester)
