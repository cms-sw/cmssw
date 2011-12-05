#!/usr/bin/env python
from sys import argv, stdout, stderr, exit
from math import *

# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from optparse import OptionParser
parser = OptionParser(usage="usage: %prog [options] bands.root graph [min max]")
parser.add_option("-l", "--level", dest="level", default=1, type="float",  help="Level to count upcrossings against")
(options, args) = parser.parse_args()
if len(args) not in [2,4]:
    parser.print_usage()
    exit(1)

file = ROOT.TFile(args[0]); 
if file == None: raise RuntimeError, "Cannot open %s" % args[0]

graph = file.Get(args[1]);
if graph == None: raise RuntimeError, "Cannot open %s" % args[0]

(mhmin, mhmax) = (0, 999)
if len(args) == 4: 
    mhmin = float(args[2])
    mhmax = float(args[3])

n = graph.GetN()
pvals = [ graph.GetY()[i] for i in range(n) if (graph.GetX()[i] >= mhmin and graph.GetX()[i] <= mhmax)]
#print "Selected ",len(pvals)," p-values."

pupcross = ROOT.ROOT.Math.normal_cdf_c(options.level, 1.0)
upcrossings = 0; pmin = pvals[-1];
for i in range(len(pvals)-1):
    if pvals[i] > pupcross and pvals[i+1] <= pupcross: 
        upcrossings += 1
    if pvals[i] < pmin: 
        pmin = pvals[i]
zmax = ROOT.ROOT.Math.normal_quantile_c(pmin, 1.0)

expfactor = exp(-0.5*(zmax**2 - options.level**2))
pglobal     = pmin + expfactor * upcrossings 
pglobal_min = pmin + expfactor * (upcrossings-sqrt(upcrossings))
pglobal_max = pmin + expfactor * (upcrossings+sqrt(upcrossings))
zglobal     = ROOT.ROOT.Math.normal_quantile_c(pglobal, 1.0)
zglobal_ehi = ROOT.ROOT.Math.normal_quantile_c(pglobal_min, 1.0) - zglobal
zglobal_elo = ROOT.ROOT.Math.normal_quantile_c(pglobal_max, 1.0) - zglobal
trials  = pglobal/pmin

print "Number of upcrossings:      Z0   = %4.2f (pval %.2e), N0 = %2d" % (options.level, pupcross, upcrossings)
print "Maximum local significance: Zmax = %4.2f (pval %.2e)" % (zmax, pmin)
print "Corrected significance:     Zglb = %4.1f %+.1f/%+.1f (pval %.2e), trials factor %d" % (zglobal, zglobal_elo, zglobal_ehi, pglobal, trials)

