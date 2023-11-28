import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test wrong process.options parameter types')

parser.add_argument("--name", help="Name of parameter", type=str)
parser.add_argument("--value", help="Value of the parameter", type=str)

<<<<<<< HEAD
argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)
=======
args = parser.parse_args()
>>>>>>> 895df58e36cff1d7dc27b1bf37aee7f604adc704

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")

process.maxEvents.input = 2
<<<<<<< HEAD

=======
#avoid type check in python to force check in C++
delattr(process.options, args.name)
>>>>>>> 895df58e36cff1d7dc27b1bf37aee7f604adc704
setattr(process.options, args.name, eval(str(args.value)))
