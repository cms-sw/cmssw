import re
from sys import argv, stdout, stderr
from optparse import OptionParser
import ROOT
parser = OptionParser()
parser.add_option("-s", "--stat",   dest="stat",    default=False, action="store_true")  # ignore systematic uncertainties to consider statistical uncertainties only
parser.add_option("-a", "--asimov", dest="asimov",  default=False, action="store_true")
parser.add_option("-c", "--compiled", dest="cexpr", default=False, action="store_true")
parser.add_option("-b", "--binary",   dest="bin",   default=False, action="store_true", help="produce a Workspace in a rootfile instead of an HLF file")
parser.add_option("-o", "--out",      dest="out",   type="string", default=None,  help="output file (if none, it will print to stdout). Required for binary mode.")

(options, args) = parser.parse_args()


file = open(args[0], "r")

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
