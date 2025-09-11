import FWCore.ParameterSet.Config as cms

import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ProcessAccelerator.')

parser.add_argument("--noConsumes", help="Do not call consumes", action="store_true")
parser.add_argument("--thing", help="Add producer and consumer for Thing", action="store_true")
parser.add_argument("--otherInt", help="Add another producer and consumer for int", action="store_true")

args = parser.parse_args()

process = cms.Process("TESTANA")
process.maxEvents.input = -1

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:getbylabel_step1.root")
)

process.intAnalyzer = cms.EDAnalyzer("edmtest::TestGetByLabelIntAnalyzer",
    src = cms.untracked.InputTag("intProduct"),
    consumes = cms.untracked.bool(True)
)

process.p = cms.Path(
    process.intAnalyzer
)

if args.thing:
    process.thingProduct = cms.EDProducer("ThingProducer")
    process.thingAnalyzer = cms.EDAnalyzer("edmtest::TestGetByLabelThingAnalyzer",
        src = cms.untracked.InputTag("thingProduct"),
        consumes = cms.untracked.bool(True)
    )
    process.p += (process.thingProduct+process.thingAnalyzer)

if args.otherInt:
    process.otherIntProduct = cms.EDProducer("IntProducer", ivalue = cms.int32(314))
    process.otherIntAnalyzer = cms.EDAnalyzer("edmtest::TestGetByLabelIntAnalyzer",
        src = cms.untracked.InputTag("otherIntProduct"),
        consumes = cms.untracked.bool(True)
    )
    process.p += (process.otherIntProduct+process.otherIntAnalyzer)

if args.noConsumes:
    process.intAnalyzer.consumes = False
    process.intAnalyzer.getExceptionCategory = cms.untracked.string("GetByLabelWithoutRegistration")

    if args.thing:
        process.thingAnalyzer.consumes = False
        process.thingAnalyzer.getExceptionCategory = cms.untracked.string("GetByLabelWithoutRegistration")
