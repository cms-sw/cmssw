#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit
from optparse import OptionParser

# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

parser = OptionParser(usage="usage: %prog [options] datacard.txt -o output \nrun with --help to get list of options")
parser.add_option("-s", "--stat",   dest="stat",    default=False, action="store_true", help="keep only statistical uncertainties, no systematics") 
parser.add_option("-c", "--compiled", dest="cexpr", default=False, action="store_true", help="use compiled expressions (not suggested)")
parser.add_option("-a", "--ascii",    dest="bin",   default=True, action="store_false", help="produce a Workspace in a rootfile in an HLF file (legacy, unsupported)")
parser.add_option("-b", "--binary",   dest="bin",   default=True, action="store_true",  help="produce a Workspace in a rootfile (default)")
parser.add_option("-o", "--out",      dest="out",   default=None,  type="string", help="output file (if none, it will print to stdout). Required for binary mode.")
parser.add_option("-v", "--verbose",  dest="verbose",  default=0,  type="int",    help="Verbosity level (0 = quiet, 1 = verbose, 2+ = more)")
parser.add_option("-m", "--mass",     dest="mass",     default=0,  type="float",  help="Higgs mass to use. Will also be written in the Workspace as RooRealVar 'MH'.")
parser.add_option("-D", "--dataset",  dest="dataname", default="data_obs",  type="string",  help="Name of the observed dataset")
parser.add_option("-L", "--LoadLibrary", dest="libs",  type="string" , action="append", help="Load these libraries")
(options, args) = parser.parse_args()
if len(args) == 0:
    parser.print_usage()
    exit(1)

file = open(args[0], "r")
options.fileName = args[0]

from HiggsAnalysis.CombinedLimit.DatacardParser import *
from HiggsAnalysis.CombinedLimit.ModelTools import *
from HiggsAnalysis.CombinedLimit.ShapeTools import *

DC = parseCard(file, options)
MB = None
if DC.hasShapes:
    MB = ShapeBuilder(DC, options)
else:
    MB = CountingModelBuilder(DC, options)

MB.doModel()
