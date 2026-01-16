import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test Refs after merge.')

parser.add_argument("--fileName", help="file to read")
parser.add_argument("--promptRead", action="store_true", default=False, help="prompt read the event products")

args = parser.parse_args()

process = cms.Process("TEST")

from IOPool.Input.modules import PoolSource
process.source = PoolSource(fileNames = [f"file:{args.fileName}"], delayReadingEventProducts = not args.promptRead)

from FWCore.Integration.modules import OtherThingAnalyzer
process.tester = OtherThingAnalyzer(other = ("d","testUserTag"))

process.e = cms.EndPath(process.tester)

