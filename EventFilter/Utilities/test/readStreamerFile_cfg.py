import FWCore.ParameterSet.Config as cms
import argparse
import sys

process = cms.Process("READ")


parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test output of GlobalEvFOutputModule')
parser.add_argument("--input", action="append", default=[], help="Input files")
parser.add_argument("--runNumber", type=int, default=1, help="expected run number")
parser.add_argument("--numEvents", type=int, default=10, help="expected number of events")
args = parser.parse_args()
if len(args.input) == 0:
    parser.error("No input files")

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(["file:"+f for f in args.input])
)

process.tester = cms.EDAnalyzer("OtherThingAnalyzer",
    other = cms.untracked.InputTag("otherThing","testUserTag")
)

rn = args.runNumber
lumi = 1
transitions = [cms.EventID(rn,0,0),cms.EventID(rn,lumi,0)]
evid = 1

for ev in range(0, args.numEvents):
    transitions.append(cms.EventID(rn,lumi,evid))
    evid += 1
transitions.append(cms.EventID(rn,lumi,0)) #end lumi
transitions.append(cms.EventID(rn,0,0)) #end run

if args.numEvents == 0:
    transitions = []

process.test = cms.EDAnalyzer("RunLumiEventChecker",
                              eventSequence = cms.untracked.VEventID(*transitions),
                              unorderedEvents = cms.untracked.bool(True)
)


process.e = cms.EndPath(process.tester+process.test)
