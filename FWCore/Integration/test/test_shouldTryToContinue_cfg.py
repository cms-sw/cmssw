import FWCore.ParameterSet.Config as cms

import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test TryToContinue exception handling.')

parser.add_argument("--indirect", help="Apply shouldTryToContinue to module dependent on module that fails. Else apply to failing module.", action="store_true")
parser.add_argument("--inRun", action = "store_true", help="Throw exception in Run")
parser.add_argument("--inLumi", action = "store_true", help="Throw exception in Lumi")

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
process.intProd = cms.EDProducer("IntProducer", ivalue = cms.int32(10))
process.dependentAnalyzer = cms.EDAnalyzer("TestFindProduct",
    inputTags = cms.untracked.VInputTag(["intProd"]),
    inputTagsNotFound = cms.untracked.VInputTag( cms.InputTag("fail")),
    expectedSum = cms.untracked.int32(30)
)

process.dependent2 = cms.EDAnalyzer("TestFindProduct",
    inputTags = cms.untracked.VInputTag(["intProd"]),
    inputTagsNotFound = cms.untracked.VInputTag( cms.InputTag("fail")),
    expectedSum = cms.untracked.int32(30)
)

process.independent = cms.EDAnalyzer("TestFindProduct",
    inputTags = cms.untracked.VInputTag(["intProd"]),
    expectedSum = cms.untracked.int32(30)
)

process.f = cms.EDFilter("IntProductFilter", label = cms.InputTag("intProd"))

if args.inRun:
    process.dependentAnalyzer.inputTagsEndRun = cms.untracked.VInputTag(cms.InputTag("fail"))
if args.inLumi:
    process.dependentAnalyzer.inputTagsEndLumi = cms.untracked.VInputTag(cms.InputTag("fail"))
if args.inRun or args.inLumi:
    process.dependentAnalyzer.expectedSum = 0
    process.dependent2.expectedSum = 0
    process.independent.expectedSum = 0

if args.indirect:
    process.options.modulesToCallForTryToContinue = [process.dependentAnalyzer.label_(), process.dependent2.label_()]
else:
    process.options.modulesToCallForTryToContinue = [process.fail.label_()]

process.p = cms.Path(process.dependentAnalyzer, cms.Task(process.fail,process.intProd))
process.p2 = cms.Path(cms.wait(process.dependent2)+process.f+process.independent)
#process.add_(cms.Service("Tracer"))
