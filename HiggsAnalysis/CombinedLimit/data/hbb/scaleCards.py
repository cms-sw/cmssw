#!/usr/bin/env python
import re, os
from sys import argv, stdout, stderr, exit
from optparse import OptionParser
from math import *

# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )
parser = OptionParser(usage="usage: %prog [options] mass \nrun with --help to get list of options")
parser.add_option("--xsbr",  dest="xsbr",   action="store_true", default=False, help="Use correct XS*BR for Higgs (if not enabled, just makes a flat copy)")
parser.add_option("--ddir",  dest="ddir",   type="string",       default=".",   help="Path to the datacards")
parser.add_option("--refmasse",  dest="refmass", type="float",  default="0",   help="Reference mass to start from (default = nearest one, left in case of ties)")
parser.add_option("--postfix",   dest="postfix", type="string", default="",    help="Postfix to add to datacard name")
parser.add_option("--flavour",   dest="flavour", type="string", default="BDT", help="flavour of datacard (vhbb_DC_ALL_<FLAVOUR>.<MASS.DECIMAL>.txt)")
(options, args) = parser.parse_args()
options.bin = True; options.stat = False
if len(args) not in [1]:
    parser.print_usage()
    exit(1)

from HiggsAnalysis.CombinedLimit.DatacardParser import *

refmasses = range(110,135+1,5)
mass = float(args[0])

if options.refmass == 0:
    options.refmass = refmasses[0]
    for m in refmasses[1:]:
        if abs(mass-m) < abs(mass-options.refmass): 
            options.refmass = m

if mass in refmasses and options.postfix == "": raise RuntimeError, "Will not overwrite the reference masses"

xsbrR = { 'WH':1.0, 'ZH':1.0 }
xsbr  = { 'WH':1.0, 'ZH':1.0 }
if options.xsbr:
    def file2map(x): 
        ret = {}; headers = []
        for x in open(x,"r"):
            cols = x.split()
            if len(cols) < 2: continue
            if "mH" in x: 
                headers = [i.strip() for i in cols[1:]]
            else:
                fields = [ float(i) for i in cols ]
                ret[fields[0]] = dict(zip(headers,fields[1:]))
        return ret
    path = os.environ['CMSSW_BASE']+"/src/HiggsAnalysis/CombinedLimit/data/";   
    whXS = file2map(path+"YR-XS-WH.txt")
    zhXS = file2map(path+"YR-XS-ZH.txt")
    br   = file2map(path+"YR-BR1.txt")
    xsbrR['WH'] = whXS[options.refmass]['XS_pb'] * br[options.refmass]['H_bb']
    xsbr ['WH'] = whXS[mass           ]['XS_pb'] * br[mass           ]['H_bb']
    xsbrR['ZH'] = zhXS[options.refmass]['XS_pb'] * br[options.refmass]['H_bb']
    xsbr ['ZH'] = zhXS[mass           ]['XS_pb'] * br[mass           ]['H_bb']
    print "Will interpolate %g from %g (XS*BR ratio: %.3f for WH, %.3f for ZH)" % (mass, options.refmass, xsbr['WH']/xsbrR['WH'], xsbr['ZH']/xsbrR['ZH'])
else:
    print "Will copy %g from %g" % (mass, options.refmass)

fileR = options.ddir+"/%g/vhbb_DC_ALL_%s.%.1f.txt" % (options.refmass, options.flavour, options.refmass)
options.fileName = fileR; options.mass = options.refmass;
DCR = parseCard(open(fileR,"r"), options)

obskeyline = DCR.bins; 
obsline    = [str(DCR.obs[b]) for b in DCR.bins]; 
cmax = 5;
keyline = []; expline = []; systlines = {}; systlines2 = {}
signals = []; backgrounds = []; shapeLines = []; 
paramSysts = {}; flatParamNuisances = {}
for (name,nf,pdf,args,errline) in DCR.systs:
    systlines[name] = [ pdf, args, errline, nf ]

for b,p,sig in DCR.keyline:
    rate = DCR.exp[b][p]
    if p == "VH":
        pTrue = "WH" if b[0] == "W" else "ZH";
        rate = rate * xsbr[pTrue]/xsbrR[pTrue]
    keyline.append( (b, p, DCR.isSignal[p]) )
    expline.append( "%.4f" % rate )

xfile = open(options.ddir+"/%g/vhbb_DC_ALL_%s%s.%.1f.txt" % (mass, options.flavour, options.postfix, mass), "w")
xfile.write(" ".join(["imax %d number of bins" % len(DCR.bins)])+"\n")
xfile.write(" ".join(["jmax *  number of processes minus 1"])+"\n")
xfile.write(" ".join(["kmax *  number of nuisance parameters"])+"\n")
xfile.write(" ".join(["-" * 130])+"\n")

cmax = max([cmax]+[len(l) for l in obskeyline]+[len(x) for x in obsline])
cfmt = "%-"+str(cmax)+"s";
xfile.write(" ".join(["bin         ", "  ".join([cfmt % x for x in obskeyline])])+"\n")
xfile.write(" ".join(["observation ", "  ".join([cfmt % x for x in obsline])])+"\n")

xfile.write(" ".join(["-" * 150])+"\n")

pidline = []; signals = []; backgrounds = []
for (b,p,s) in keyline:
    if s:
        if p not in signals: signals.append(p)
        pidline.append(-len(DCR.signals)+signals.index(p)+1)
    else:
        if p not in backgrounds: backgrounds.append(p)
        pidline.append(1+backgrounds.index(p))
cmax = max([cmax]+[max(len(p),len(b)) for p,b,s in keyline]+[len(e) for e in expline])
hmax = max([10] + [len("%-12s  %s %s" % (l,p,a)) for l,(p,a,e,nf) in systlines.items()])
cfmt  = "%-"+str(cmax)+"s"; hfmt = "%-"+str(hmax)+"s  ";
xfile.write(" ".join([hfmt % "bin",     "  ".join([cfmt % p for p,b,s in keyline])])+"\n")
xfile.write(" ".join([hfmt % "process", "  ".join([cfmt % b for p,b,s in keyline])])+"\n")
xfile.write(" ".join([hfmt % "process", "  ".join([cfmt % x for x in pidline])])+"\n")
xfile.write(" ".join([hfmt % "rate",    "  ".join([cfmt % x for x in expline])])+"\n")
xfile.write(" ".join(["-" * 150])+"\n")
sysnamesSorted = systlines.keys(); sysnamesSorted.sort()
for name in sysnamesSorted:
    (pdf,pdfargs,effect,nofloat) = systlines[name]
    if nofloat: name += "[nofloat]"
    systline = []
    for b,p,s in keyline:
        try:
            systline.append(effect[b][p] if (effect[b][p] != 1.0 or pdf != 'lnN') else "-")
        except KeyError:
            systline.append("-");
    xfile.write(" ".join([hfmt % ("%-28s   %s  %s" % (name, pdf, " ".join(pdfargs))), "  ".join([cfmt % x for x in systline])])+"\n")

