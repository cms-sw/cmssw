#!/usr/bin/env python
import re
import os.path
from math import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-c", "--channel", type="string", dest="channel", default=None, help="Channel to dump")
parser.add_option("-N", "--norm-only",   dest="norm",    default=False, action="store_true", help="Include only normalization uncertainties, not shape ones") 
parser.add_option("-f", "--format", type="string", dest="format", default="%8.3f +/- %6.3f", help="Format for output number")
parser.add_option("--xs", "--exclude-syst", type="string", dest="excludeSyst", default=[], action="append", help="Systematic to exclude (regexp)")
parser.add_option("-m", "--mass",     dest="mass",     default=0,  type="float",  help="Higgs mass to use. Will also be written in the Workspace as RooRealVar 'MH'.")
parser.add_option("-D", "--dataset",  dest="dataname", default="data_obs",  type="string",  help="Name of the observed dataset")
(options, args) = parser.parse_args()
options.stat = False
options.bin = True # fake that is a binary output, so that we parse shape lines
options.out = "tmp.root"
options.fileName = args[0]
options.cexpr = False
options.fixpars = False
options.libs = []
options.verbose = 0
options.poisson = 0
options.nuisancesToExclude = []
options.noJMax = True

# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
import sys
sys.argv = [ '-b-']
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit.so")

from HiggsAnalysis.CombinedLimit.DatacardParser import *
from HiggsAnalysis.CombinedLimit.ShapeTools     import *
file = open(args[0], "r")
DC = parseCard(file, options)
if not DC.hasShapes: DC.hasShapes = True
MB = ShapeBuilder(DC, options)
for b in DC.bins:
    print " ============= ", b , "===================="
    if options.channel != None and (options.channel != b): continue
    exps = {}
    for (p,e) in DC.exp[b].items(): # so that we get only self.DC.processes contributing to this bin
        exps[p] = [ e, [] ]
    for (lsyst,nofloat,pdf,pdfargs,errline) in DC.systs:
        if pdf in ('param', 'flatParam'): continue
        # begin skip systematics
        skipme = False
        for xs in options.excludeSyst:
            if re.search(xs, lsyst): 
                skipme = True
        if skipme: continue
        # end skip systematics
        for p in DC.exp[b].keys(): # so that we get only self.DC.processes contributing to this bin
            if errline[b][p] == 0: continue
            if pdf == 'gmN':
                exps[p][1].append(1/sqrt(pdfargs[0]+1));
            elif pdf == 'gmM':
                exps[p][1].append(errline[b][p]);
            elif type(errline[b][p]) == list: 
                kmax = max(errline[b][p][0], errline[b][p][1], 1.0/errline[b][p][0], 1.0/errline[b][p][1]);
                exps[p][1].append(kmax-1.);
            elif pdf == 'lnN':
                exps[p][1].append(max(errline[b][p], 1.0/errline[b][p])-1.);
            elif ("shape" in pdf) and not options.norm:
                s0 = MB.getShape(b,p)
                sUp   = MB.getShape(b,p,lsyst+"Up")
                sDown = MB.getShape(b,p,lsyst+"Down")
                if (s0.InheritsFrom("TH1")):
                    ratios = [sUp.Integral()/s0.Integral(), sDown.Integral()/s0.Integral()]
                    ratios += [1/ratios[0], 1/ratios[1]]
                    exps[p][1].append(max(ratios) - 1)
    procs = DC.exp[b].keys(); procs.sort()
    fmt = ("%%-%ds " % max([len(p) for p in procs]))+"  "+options.format;
    for p in procs:
        relunc = sqrt(sum([x*x for x in exps[p][1]]))
        print fmt % (p, exps[p][0], exps[p][0]*relunc)
    
