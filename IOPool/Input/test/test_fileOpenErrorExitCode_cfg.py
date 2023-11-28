import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description="Test FileOpenErrorExitCode")
parser.add_argument("--input", type=str, default=[], nargs="*", help="Optional list of input files")
<<<<<<< HEAD

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)
=======
args = parser.parse_args()
>>>>>>> 895df58e36cff1d7dc27b1bf37aee7f604adc704

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("/store/"+x for x in args.input)
)
