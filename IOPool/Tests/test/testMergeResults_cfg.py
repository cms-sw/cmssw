import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test merge results using TestMergeResults or TestMergeResultsOutputModule.')

parser.add_argument("--inputFiles", nargs="+", help="input file names", required=True)
parser.add_argument("--expectedBeginRunProd", nargs="+", type=int, default=[], help="expected values for beginRun products (sets of 3: Thing, ThingWithMerge, ThingWithIsEqual)")
parser.add_argument("--expectedEndRunProd", nargs="+", type=int, default=[], help="expected values for endRun products (sets of 3)")
parser.add_argument("--expectedBeginLumiProd", nargs="+", type=int, default=[], help="expected values for beginLuminosityBlock products (sets of 3)")
parser.add_argument("--expectedEndLumiProd", nargs="+", type=int, default=[], help="expected values for endLuminosityBlock products (sets of 3)")
parser.add_argument("--expectedBeginRunNew", nargs="+", type=int, default=[], help="expected values for latest process beginRun products (sets of 3)")
parser.add_argument("--expectedEndRunNew", nargs="+", type=int, default=[], help="expected values for latest process endRun products (sets of 3)")
parser.add_argument("--expectedBeginLumiNew", nargs="+", type=int, default=[], help="expected values for latest process beginLuminosityBlock products (sets of 3)")
parser.add_argument("--expectedEndLumiNew", nargs="+", type=int, default=[], help="expected values for latest process endLuminosityBlock products (sets of 3)")
parser.add_argument("--useOutputModule", action="store_true", help="use TestMergeResultsOutputModule instead of TestMergeResults")
parser.add_argument("--verbose", action="store_true", help="enable verbose output")

args = parser.parse_args()

process = cms.Process("TEST")

from IOPool.Input.modules import PoolSource
process.source = PoolSource(fileNames = [f"file:{f}" for f in args.inputFiles])

if args.useOutputModule:
    from FWCore.Framework.modules import TestMergeResultsOutputModule
    process.test = TestMergeResultsOutputModule(
        expectedBeginRunProd = args.expectedBeginRunProd,
        expectedEndRunProd = args.expectedEndRunProd,
        expectedBeginLumiProd = args.expectedBeginLumiProd,
        expectedEndLumiProd = args.expectedEndLumiProd,
        expectedBeginRunNew = args.expectedBeginRunNew,
        expectedEndRunNew = args.expectedEndRunNew,
        expectedBeginLumiNew = args.expectedBeginLumiNew,
        expectedEndLumiNew = args.expectedEndLumiNew,
        verbose = args.verbose
    )
    process.e = cms.EndPath(process.test)
else:
    from FWCore.Integration.modules import TestMergeResults
    process.test = TestMergeResults(
        expectedBeginRunProd = args.expectedBeginRunProd,
        expectedEndRunProd = args.expectedEndRunProd,
        expectedBeginLumiProd = args.expectedBeginLumiProd,
        expectedEndLumiProd = args.expectedEndLumiProd,
        expectedBeginRunNew = args.expectedBeginRunNew,
        expectedEndRunNew = args.expectedEndRunNew,
        expectedBeginLumiNew = args.expectedBeginLumiNew,
        expectedEndLumiNew = args.expectedEndLumiNew,
        verbose = args.verbose
    )
    process.e = cms.EndPath(process.test)

#process.add_(cms.Service("Tracer"))