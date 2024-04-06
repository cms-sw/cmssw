import FWCore.ParameterSet.Config as cms

import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test TryToContinue exception handling.')

parser.add_argument("--useTask", help="Put failing module in a Task", action="store_true")
parser.add_argument("--inRun", help="throw exception in begin run", action="store_true")
parser.add_argument("--inLumi", help="throw exception in begin lumi", action="store_true")

args = parser.parse_args()

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.options.TryToContinue = ['NotFound']
process.maxEvents.input = 3

if args.inRun:
    process.fail = cms.EDProducer("edmtest::FailingInRunProducer")
elif args.inLumi:
    process.fail = cms.EDProducer("edmtest::FailingInLumiProducer")
else:
    process.fail = cms.EDProducer("FailingProducer")

process.shouldRun1 = cms.EDAnalyzer("edmtest::global::StreamIntAnalyzer", transitions = cms.int32(4+3), nLumis = cms.untracked.uint32(1), verbose = cms.untracked.bool(False))
process.shouldRun2 = cms.EDAnalyzer("edmtest::global::StreamIntAnalyzer", transitions = cms.int32(4+3), nLumis = cms.untracked.uint32(1), verbose = cms.untracked.bool(False))
process.shouldNotRun = cms.EDAnalyzer("edmtest::global::StreamIntAnalyzer", transitions = cms.int32(4), nLumis = cms.untracked.uint32(1), verbose = cms.untracked.bool(False))
process.dependentFilter = cms.EDFilter("IntProductFilter",
   label = cms.InputTag("fail"),
   threshold = cms.int32(0),
   shouldProduce = cms.bool(False)
)

process.intProd = cms.EDProducer("IntProducer", ivalue = cms.int32(10))
process.addInts = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProd"))

process.dependentAnalyzer = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("fail"), cms.InputTag("addInts") ),
  expectedSum = cms.untracked.int32(0)
)

process.independentAnalyzer = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("addInts") ),
  expectedSum = cms.untracked.int32(30)
)

if args.inRun:
    process.independentAnalyzer.expectedSum = 0
    process.shouldRun1.transitions=2
    process.shouldRun2.transitions=2
    process.shouldNotRun.transitions=2
    process.shouldRun1.nLumis=0
    process.shouldRun2.nLumis=0
    process.shouldNotRun.nLumis=0

if args.inLumi:
    process.independentAnalyzer.expectedSum = 0
    process.shouldRun1.transitions=4
    process.shouldRun2.transitions=4
    process.shouldNotRun.transitions=4
    process.shouldRun1.nLumis=0
    process.shouldRun2.nLumis=0
    process.shouldNotRun.nLumis=0

process.seq = cms.Sequence()
process.t = cms.Task(process.intProd,process.addInts)
if args.useTask:
    process.t.add(process.fail)
else:
    process.seq = cms.Sequence(process.fail)
process.errorPath = cms.Path(process.seq+process.shouldRun1+process.dependentFilter+process.shouldNotRun,process.t)
process.goodPath = cms.Path(process.shouldRun2)


process.errorEndPath = cms.EndPath(process.dependentAnalyzer)
process.goodEndPath = cms.EndPath(process.independentAnalyzer)
