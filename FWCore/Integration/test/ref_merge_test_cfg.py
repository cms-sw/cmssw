import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test Refs after merge.')

parser.add_argument("--fileName", help="file to read")

args = parser.parse_args()

process = cms.Process("TEST")

from IOPool.Input.modules import PoolSource
process.source = PoolSource(fileNames = [f"file:{args.fileName}"])

from FWCore.Integration.modules import OtherThingAnalyzer
process.tester = OtherThingAnalyzer(other = ("d","testUserTag"))

process.e = cms.EndPath(process.tester)

