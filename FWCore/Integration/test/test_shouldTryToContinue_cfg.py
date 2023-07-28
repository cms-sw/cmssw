import FWCore.ParameterSet.Config as cms

import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test TryToContinue exception handling.')

parser.add_argument("--indirect", help="Apply shouldTryToContinue to module dependent on module that fails. Else apply to failing module.", action="store_true")

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.options.TryToContinue = ['NotFound']
process.maxEvents.input = 3

process.fail = cms.EDProducer("FailingProducer")
process.intProd = cms.EDProducer("IntProducer", ivalue = cms.int32(10))
process.dependentAnalyzer = cms.EDAnalyzer("TestFindProduct",
    inputTags = cms.untracked.VInputTag(["intProd"]),
    inputTagsNotFound = cms.untracked.VInputTag( cms.InputTag("fail")),
    expectedSum = cms.untracked.int32(30)
)

if args.indirect:
    process.dependentAnalyzer.shouldTryToContinue()
else:
    process.fail.shouldTryToContinue()

process.p = cms.Path(process.dependentAnalyzer, cms.Task(process.fail,process.intProd))

process.add_(cms.Service("Tracer"))
