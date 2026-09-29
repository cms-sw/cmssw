import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Merge files using PoolSource and write with PoolOutputModule.')

parser.add_argument("--inputFiles", nargs="+", help="input file names", required=True)
parser.add_argument("--outputFile", help="name of output file", required=True)

args = parser.parse_args()

process = cms.Process("MERGE")

from IOPool.Input.modules import PoolSource
process.source = PoolSource(fileNames = [f"file:{f}" for f in args.inputFiles])

from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(fileName = args.outputFile)

process.o = cms.EndPath(process.out)
