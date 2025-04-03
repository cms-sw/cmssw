import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description="Test FileOpenErrorExitCode")
parser.add_argument("--input", type=str, default=[], nargs="*", help="Optional list of input files")

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("/store/"+x for x in args.input)
)
