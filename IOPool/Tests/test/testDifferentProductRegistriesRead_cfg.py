import FWCore.ParameterSet.Config as cms
import argparse

parser = argparse.ArgumentParser(description="cmsRun config for reading two files with different product registries")
parser.add_argument("--inputFiles", nargs=2, required=True, help="Input files for PoolSource (two required)")
args, unknown = parser.parse_known_args()

process = cms.Process("READ")

from IOPool.Input.modules import PoolSource
process.source = PoolSource(
    fileNames = cms.untracked.vstring(*args.inputFiles)
)

from FWCore.Modules.modules import AsciiOutputModule
process.asciiOut = AsciiOutputModule()
process.outpath = cms.EndPath(process.asciiOut)
