import FWCore.ParameterSet.Config as cms

import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test TryToContinue exception handling.')

parser.add_argument("--useTask", help="Have filter succeed", action="store_true")

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.options.TryToContinue = ['NotFound']
process.maxEvents.input = 3

process.fail = cms.EDProducer("FailingProducer")

process.shouldRun1 = cms.EDAnalyzer("edmtest::global::StreamIntAnalyzer", transitions = cms.int32(6+3), verbose = cms.untracked.bool(False))
process.shouldRun2 = cms.EDAnalyzer("edmtest::global::StreamIntAnalyzer", transitions = cms.int32(6+3), verbose = cms.untracked.bool(False))
process.shouldNotRun = cms.EDAnalyzer("edmtest::global::StreamIntAnalyzer", transitions = cms.int32(6), verbose = cms.untracked.bool(False))
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


