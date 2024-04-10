import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ConditionalTasks.')

parser.add_argument("--filterSucceeds", help="Have filter succeed", action="store_true")
parser.add_argument("--testView", help="Get data via a view", action="store_true")

args = parser.parse_args()

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

# ensure the printout is done only once
process.maxEvents.input = 4
process.options.numberOfThreads = 4

process.MessageLogger.files.conditionaltasks_nonconsumed = dict()
process.MessageLogger.files.conditionaltasks_nonconsumed.default = dict(limit=0)
process.MessageLogger.files.conditionaltasks_nonconsumed.NonConsumedConditionalModules = dict(limit=100)

process.a = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.b = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag(cms.InputTag("a")))

process.f1 = cms.EDFilter("IntProductFilter", label = cms.InputTag("b"))

process.c = cms.EDProducer("IntProducer", ivalue = cms.int32(2))
process.d = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag(cms.InputTag("c")))
process.e = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag(cms.InputTag("d")))

process.nonconsumed = process.a.clone()
process.nonconsumed2 = process.b.clone(
    labels = ["nonconsumed"]
)
process.nonconsumedConditionalTask = cms.ConditionalTask(
    process.nonconsumed,
    process.nonconsumed2,
)

process.consumedInOnePath = process.a.clone()
process.nonconsumedConditionalTask2 = cms.ConditionalTask(
    process.nonconsumed,
    process.consumedInOnePath
)

process.explicitlyInDifferentPath = process.a.clone()
process.consumedInUnrelatedPath = process.a.clone()
process.nonconsumedConditionalTask3 = cms.ConditionalTask(
    process.nonconsumed,
    process.explicitlyInDifferentPath,
    process.consumedInUnrelatedPath
)
process.nonconsumedTask3 = cms.Task(
    process.consumedInUnrelatedPath
)

process.prodOnPath = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag(cms.InputTag("d"), cms.InputTag("e")))
process.prodOnPath2 = process.prodOnPath.clone(
    labels = ["consumedInOnePath"]
)
process.prodOnPath3 = process.prodOnPath.clone(
    labels = ["consumedInUnrelatedPath"]
)

if args.filterSucceeds:
    threshold = 1
else:
    threshold = 3

process.f2 = cms.EDFilter("IntProductFilter", label = cms.InputTag("e"), threshold = cms.int32(threshold))

if args.testView:
  process.f3 = cms.EDAnalyzer("SimpleViewAnalyzer",
    label = cms.untracked.InputTag("f"),
    sizeMustMatch = cms.untracked.uint32(10),
    checkSize = cms.untracked.bool(False)
  )
  process.f = cms.EDProducer("OVSimpleProducer", size = cms.int32(10))
  producttype = "edmtestSimplesOwned"
else:
  process.f= cms.EDProducer("IntProducer", ivalue = cms.int32(3))
  process.f3 = cms.EDFilter("IntProductFilter", label = cms.InputTag("f"))
  producttype = "edmtestIntProduct"

process.p = cms.Path(process.f1+process.prodOnPath+process.f2+process.f3, cms.ConditionalTask(process.a, process.b, process.c, process.d, process.e, process.f, process.nonconsumedConditionalTask, process.nonconsumedConditionalTask2))

process.p2 = cms.Path(process.prodOnPath2, process.nonconsumedConditionalTask2, process.nonconsumedConditionalTask3)

process.p3 = cms.Path(process.explicitlyInDifferentPath)

process.p4 = cms.Path(process.prodOnPath3, process.nonconsumedTask3)

process.tst = cms.EDAnalyzer("IntTestAnalyzer", moduleLabel = cms.untracked.InputTag("f"), valueMustMatch = cms.untracked.int32(3), 
                                                       valueMustBeMissing = cms.untracked.bool(not args.filterSucceeds))

process.nonconsumedConsumer = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag(cms.InputTag("nonconsumed")))

process.endp = cms.EndPath(process.tst+process.nonconsumedConsumer)
