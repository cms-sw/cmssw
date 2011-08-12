import re
from sys import argv, stdout, stderr, exit

# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

if len(argv) == 0: raise RuntimeError, "Usage: hwwNormToText.py mlfit.root";

file = ROOT.TFile.Open(argv[1]);
fit_s = file.Get("norm_fit_s")
fit_b = file.Get("norm_fit_b")
if fit_s == None: raise RuntimeError, "Missing fit_s in %s. Did you run MaxLikelihoodFit with --saveNorm?" % file;
if fit_b == None: raise RuntimeError, "Missing fit_b in %s. Did you run MaxLikelihoodFit with --saveNorm?" % file;

iter = fit_s.createIterator()
while True:
    norm_s = iter.Next()
    if norm_s == None: break;
    norm_b = fit_b.find(norm_s.GetName())
    m = re.match(r"n_exp_bin(\w+)_proc_(\w+)", norm_s.GetName());
    if m == None: raise RuntimeError, "Non-conforming object name %s" % norm_s.GetName()
    if norm_b == None: raise RuntimeError, "Missing normalization %s for background fit" % norm_s.GetName()
    print "%-30s %-30s %7.3f %7.3f" % (m.group(1), m.group(2), norm_s.getVal(), norm_b.getVal())
