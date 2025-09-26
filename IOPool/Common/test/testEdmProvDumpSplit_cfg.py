import FWCore.ParameterSet.Config as cms
import argparse

parser = argparse.ArgumentParser(description="Merge files in edmProvDump test")
parser.add_argument("--process", default="SPLIT", help="Process name")
parser.add_argument("--file", action="append", type=str, help="Input files")
parser.add_argument("--lumi", type=int, help="If set, process only this LuminosityBlock")
parser.add_argument("--output", default="merged_files.root", help="Output file name")
parser.add_argument("--ivalue", type=int, default=11, help="Value for one tracked parameter")
parser.add_argument("--version", type=str, help="CMSSW version to be used in the ProcessHistory (default is unset")
args = parser.parse_args()

process = cms.Process(args.process)
if args.version:
    process._specialOverrideReleaseVersionOnlyForTesting(args.version)

from IOPool.Input.modules import PoolSource
process.source = PoolSource(
    fileNames = ["file:"+f for f in args.file],
)
if args.lumi:
    process.source.lumisToProcess = [cms.LuminosityBlockRange(1,args.lumi, 1,args.lumi)]

from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(fileName = args.output)

from FWCore.Framework.modules import IntVectorProducer
process.intVectorProducer = IntVectorProducer(
  count = 9,
  ivalue = args.ivalue
)

process.p = cms.Path(process.intVectorProducer)
process.endp = cms.EndPath(process.out)
