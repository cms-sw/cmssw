import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test module transform.')

#do not call transform
#on Path/off Path

parser.add_argument("--onPath", help="Put transform module on the Path", action="store_true")
parser.add_argument("--noTransform", help="do not consume transform", action="store_true")
parser.add_argument("--stream", help="use stream module", action="store_true")
parser.add_argument("--noPut", help="do not put data used by transform", action="store_true")
parser.add_argument("--addTracer", help="add Tracer service", action="store_true")
parser.add_argument("--async_", help="use asynchronous module", action="store_true")
parser.add_argument("--exception", help="Make module consumed by transformer to throw an exception", action="store_true")

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)


process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 4

process.start = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
if args.exception:
  process.start = cms.EDProducer("FailingProducer")
if args.stream:
  if args.async_:
    process.t = cms.EDProducer("TransformAsyncIntStreamProducer", get = cms.InputTag("start"), offset = cms.uint32(1), checkTransformNotCalled = cms.untracked.bool(False))
  else:
    process.t = cms.EDProducer("TransformIntStreamProducer", get = cms.InputTag("start"), offset = cms.uint32(1), checkTransformNotCalled = cms.untracked.bool(False))
else:
  if args.async_:
    process.t = cms.EDProducer("TransformAsyncIntProducer", get = cms.InputTag("start"), offset = cms.uint32(1), checkTransformNotCalled = cms.untracked.bool(False), noPut = cms.bool(args.noPut))
  else:
    process.t = cms.EDProducer("TransformIntProducer", get = cms.InputTag("start"), offset = cms.uint32(1), checkTransformNotCalled = cms.untracked.bool(False), noPut = cms.bool(args.noPut))

process.tester = cms.EDAnalyzer("IntTestAnalyzer",
                                moduleLabel = cms.untracked.InputTag("t","transform"),
                                valueMustMatch = cms.untracked.int32(3))

if args.noTransform:
    process.tester.moduleLabel = "t"
    process.tester.valueMustMatch = 2
    process.t.checkTransformNotCalled = True

process.nonConsumed = process.t.clone()

if args.onPath:
    process.p = cms.Path(process.t+process.tester, cms.Task(process.start, process.nonConsumed))
else:
    process.p = cms.Path(process.tester, cms.Task(process.start, process.t, process.nonConsumed))

if args.addTracer:
    process.add_(cms.Service("Tracer"))
