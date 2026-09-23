import FWCore.ParameterSet.Config as cms
import argparse

parser = argparse.ArgumentParser(description='Test edmProvDump')
parser.add_argument("--version", type=str, help="CMSSW version to be used in the ProcessHistory (default is unset")
args = parser.parse_args()

process = cms.Process("TEST")
if args.version:
    process._specialOverrideReleaseVersionOnlyForTesting(args.version)

from FWCore.Modules.modules import EmptySource
process.source = EmptySource()

process.maxEvents.input = 1

from FWCore.Integration.testProducerWithPsetDescEmpty_cfi import *
process.testProducerWithPsetDesc = testProducerWithPsetDesc
process.t = cms.Path(process.testProducerWithPsetDesc)


from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(
    fileName = "testEdmProvDumpPSet.root"
)
process.ep = cms.EndPath(process.out)
