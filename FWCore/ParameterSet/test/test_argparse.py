import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser(description='Test argparse')
parser.add_argument("--maxEvents", help="max events to process", type=int, default=1)
# same as an edmConfigDump argument
parser.add_argument("-o", "--output", help="output filename", type=str, default=None)
# change parameter of tracked module
parser.add_argument("-i", "--intprod", help="int value to produce", type=int, default=1)
args = parser.parse_args()

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")

process.maxEvents.input = args.maxEvents

process.m1a = cms.EDProducer("IntProducer",
    ivalue = cms.int32(args.intprod)
)
process.p1 = cms.Path(process.m1a)

if args.output is not None:
    process.testout1 = cms.OutputModule("TestOutputModule",
        name = cms.string(args.output),
    )
    process.e1 = cms.EndPath(process.testout1)
