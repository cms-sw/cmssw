import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test Refs after merge.')

parser.add_argument("--fileNames", nargs='+', help="files to read")
parser.add_argument("--nEventsToFail", help='number of events to fail before actually reading file', type=int)
parser.add_argument("--thing2Dropped", help="tell `other` to expect thing2 to be missing", action='store_true')
parser.add_argument("--thing2NotRun", help="the thing2 module was never run", action="store_true")
parser.add_argument("--other2Run", help="the other2 module was run", action="store_true")
parser.add_argument("--thing2DependsOnThing1", help="in previous job thing2 depends on thing1", action='store_true')
args = parser.parse_args()

print(args)
process = cms.Process("TEST")

from IOPool.Input.modules import PoolSource
process.source = PoolSource(fileNames = [f"file:{n}" for n in args.fileNames])

from FWCore.Integration.modules import TestFindProduct, TestParentage

process.filt = cms.EDFilter("TestFilterModule", acceptValue = cms.untracked.int32(args.nEventsToFail))
getTags=[]
missingTags=[]
if args.thing2Dropped or args.thing2NotRun:
    missingTags=['thing2']
else:
    getTags=['thing2']
process.other = TestFindProduct(getByTokenFirst=True, inputTags=getTags, inputTagsNotFound=missingTags)
expectedAncestors = []
if args.thing2NotRun:
    expectedAncestors = ['thing1']
else:
    if args.thing2DependsOnThing1:
        expectedAncestors = ['thing2', 'thing1']
    else:
        expectedAncestors = ['thing2']

process.e = cms.Path(~process.filt+process.other)
if args.other2Run:
    process.parentage = TestParentage(inputTag='other2', expectedAncestors=expectedAncestors)
    process.e +=process.parentage

