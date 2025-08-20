import FWCore.ParameterSet.Config as cms
import argparse

parser = argparse.ArgumentParser(description='Create files for reduced ProcessHistory test')
parser.add_argument("file1", type=str, help="First file to merge")
parser.add_argument("file2", type=str, help="Second file to merge")
parser.add_argument("--output", default="merged_files.root", help="Output file name")
parser.add_argument("--bypassVersionCheck", action="store_true", help="Bypass version check")

args = parser.parse_args()

process = cms.Process("MERGETWOFILES")

from IOPool.Input.modules import PoolSource
process.source = PoolSource(
    fileNames = ["file:"+args.file1,"file:"+args.file2],
    duplicateCheckMode = "noDuplicateCheck",
    bypassVersionCheck = args.bypassVersionCheck,
)

from FWCore.Integration.modules import ThingWithMergeProducer
process.thingWithMergeProducer = ThingWithMergeProducer()

from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(fileName = args.output)

process.p = cms.Path(process.thingWithMergeProducer)

process.t = cms.EndPath(process.out)
