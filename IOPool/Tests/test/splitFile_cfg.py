import FWCore.ParameterSet.Config as cms
import argparse

parser = argparse.ArgumentParser(prog='splitFile_cfg.py', description='Split a file by events using PoolSource and PoolOutputModule.')

parser.add_argument('--inputFile', type=str, required=True, help='Input file name')
parser.add_argument('--outputFile', type=str, required=True, help='Output file name')
parser.add_argument('--skipEvents', type=int, default=0, help='Number of events to skip')
parser.add_argument('--maxEvents', type=int, required=True, help='Number of events to process')

args = parser.parse_args()

process = cms.Process('SPLIT')

from IOPool.Input.modules import PoolSource
process.source = PoolSource(
    fileNames = [f'file:{args.inputFile}'],
    skipEvents = args.skipEvents
)

process.maxEvents.input = args.maxEvents

from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(
    fileName = args.outputFile
)

process.ep = cms.EndPath(process.out)
