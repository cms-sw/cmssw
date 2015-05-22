import re
from sys import argv, stdout, stderr, exit
from math import *

# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

if len(argv) == 0: raise RuntimeError, "Usage: mlfitNormsToText.py [ -u ] mlfit.root";

errors = False
if len(argv) > 2 and argv[1] == "-u": 
    errors = True
    argv[1] = argv[2];
file = ROOT.TFile.Open(argv[1]);
prefit = file.Get("norm_prefit")
fit_s = file.Get("norm_fit_s")
fit_b = file.Get("norm_fit_b")
#if prefit == None: raise RuntimeError, "Missing fit_s in %s. Did you run MaxLikelihoodFit with --saveNorm?" % file;
if fit_s  == None: raise RuntimeError, "Missing fit_s in %s. Did you run MaxLikelihoodFit with --saveNorm?" % file;
if fit_b  == None: raise RuntimeError, "Missing fit_b in %s. Did you run MaxLikelihoodFit with --saveNorm?" % file;

iter = fit_s.createIterator()
norms={}
while True:
    norm_s = iter.Next()
    if norm_s == None: break;
    norm_b = fit_b.find(norm_s.GetName())
    norm_p = prefit.find(norm_s.GetName()) if prefit else None
    m = re.match(r"(\w+)/(\w+)", norm_s.GetName());
    if m == None: m = re.match(r"n_exp_(?:final_)?(?:bin)+(\w+)_proc_(\w+)", norm_s.GetName());
    if m == None: raise RuntimeError, "Non-conforming object name %s" % norm_s.GetName()
    if norm_b == None: raise RuntimeError, "Missing normalization %s for background fit" % norm_s.GetName()
    bin = m.group(1)
    for X in "bin", "BCat_MVA": bin = bin.replace(X,"")
    if bin not in norms: norms[bin] = {}
    if prefit and norm_p:
        norms[bin][m.group(2)] = [ norm_p.getVal(), norm_p.getError(), norm_s.getVal(), norm_s.getError(), norm_b.getVal(), norm_b.getError() ]

def pp(bin,samples,index=4):
    global norms,errors
    tot, tote = 0, 0
    for s in samples.split():
        if s not in norms[bin]: continue
        tot  += norms[bin][s][index+0]
        tote += norms[bin][s][index+1]**2 if "ttH" not in samples else norms[bin][s][1]
    if tot == 0:
        return "-"
    if "ttH" in samples: tote = tote**2
    if errors:
        if "ql" in bin:
            return "$%4.2f \pm %4.2f$ " % (tot, sqrt(tote))
        else:
            return "$%4.1f \pm %4.1f$ " % (tot, sqrt(tote))
    else:
        return "%4.1f " % (tot)

def doLine(x,p,hline=False): 
    bins = [ "ttH_2lss_mumu", "ttH_2lss_ee", "ttH_2lss_em", "tl", "ql" ]
    print "%-30s " % x,
    print " ".join(["& %-18s " % pp(b,p) for b in bins]),
    print r" \\ \hline" if hline else r" \\"

signals     = "ttH_hww ttH_hzz ttH_htt".split()
backgrounds = [ "TTW", "TTZ TTGStar", "TTWW", "TTG", "WZ", "ZZ", "VVV TBZ WWqq WWDPI", "FR_data TT", "QF_data"]
doLine(r"$\ttH$, $\PH\to\PW\PW$     ", "ttH_hww")
doLine(r"$\ttH$, $\PH\to\Z\Z$       ", "ttH_hzz")
doLine(r"$\ttH$, $\PH\to\Pgt\Pgt$   ", "ttH_htt", hline=True)
doLine(r"$\ttbar\,\PW$              ", "TTW")
doLine(r"$\ttbar\,\Z\!/\!\gamma^*$  ", "TTZ TTGStar")
doLine(r"$\ttbar\,\PW\PW$           ", "TTWW")
doLine(r"$\ttbar\,\gamma$           ", "TTG", hline=True)
doLine(r"$\PW\Z$                    ", "WZ")
doLine(r"$\Z\Z$                     ", "ZZ")
doLine(r"rare SM bkg.              ", "VVV TBZ WWqq WWDPI", hline=True)
doLine(r"non-prompt ", "FR_data TT")
doLine(r"charge flip ", "QF_data", hline=True)
doLine(r"all signals ", " ".join(signals))
doLine(r"all backgrounds ", " ".join(backgrounds), hline=True)


