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
parser = OptionParser(usage="usage: %prog [options] [mass1 mass2] mass3 > datacard3.txt \nrun with --help to get list of options")
parser.add_option("--jets",  dest="jets",   type="int",    default=0,  help="Jet bin");
parser.add_option("--xsbr",  dest="xsbr",   action="store_true", default=False, help="Use correct XS*BR for Higgs")
#parser.add_option("--doeff", dest="doeff",  action="store_true", default=False, help="Include interpolation of efficiency")
#parser.add_option("--nobg",  dest="nobg",   action="store_true", default=False, help="No interpolation of background normalizations")
parser.add_option("--log",   dest="log",    action="store_true", default=False, help="Use log-scale interpolation for yields (default is linear)")
parser.add_option("--ddir",  dest="ddir", type="string", default=".", help="Path to the datacards")
parser.add_option("--refmasses",  dest="refmasses", type="string",  default="hww.masses.txt", help="File containing the reference masses between which to interpolate (relative to --options.ddir)")
parser.add_option("--postfix",    dest="postfix",   type="string",  default="",               help="Postfix to add to datacard name")
(options, args) = parser.parse_args()
options.bin = True; options.stat = False
if len(args) not in [1,3]:
    parser.print_usage()
    exit(1)

from HiggsAnalysis.CombinedLimit.DatacardParser import *

refmasses = [ int(line) for line in open(options.ddir+"/"+options.refmasses,"r") ]

if len(args) == 1:
    mass = float(args[0])
    mass1 = max([m for m in refmasses if m <= mass])
    mass2 = min([m for m in refmasses if m >= mass])
else:
    mass1 = int(args[0])
    mass2 = int(args[1])
    mass  = float(args[2])

if mass in refmasses and options.postfix == "": raise RuntimeError, "Will not overwrite the reference masses"

## Make sure mass1 is always the closest (and pick the worse one in case of a tie)
dm1 = abs(mass1 - mass)
dm2 = abs(mass2 - mass)
if (dm2 < dm1) or (dm2 == dm1 and abs(mass1 - 164) < abs(mass2 - 164)):
    (mass1, mass2) = (mass2, mass1)


xsbr1 = { 'ggH':1.0, 'qqH':1.0 }
xsbr2 = { 'ggH':1.0, 'qqH':1.0 }
xsbr  = { 'ggH':1.0, 'qqH':1.0 }
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
    ggXS = file2map(path+"YR-XS-ggH.txt")
    qqXS = file2map(path+"YR-XS-vbfH.txt")
    br   = file2map(path+"YR-BR3.txt")
    # create points at 450, 550 by interpolation
    for M in (450,550):
        ggXS[M] = dict([ (key, 0.5*(ggXS[M+10][key] + ggXS[M-10][key])) for key in ggXS[M+10].iterkeys() ])
        qqXS[M] = dict([ (key, 0.5*(qqXS[M+10][key] + qqXS[M-10][key])) for key in qqXS[M+10].iterkeys() ])
        br[M] = dict([ (key, 0.5*(br[M+10][key] + br[M-10][key])) for key in br[M+10].iterkeys() ])
    xsbr1['ggH'] = ggXS[mass1]['XS_pb'] * br[mass1]['H_evmv']
    xsbr2['ggH'] = ggXS[mass2]['XS_pb'] * br[mass2]['H_evmv']
    xsbr ['ggH'] = ggXS[mass ]['XS_pb'] * br[mass ]['H_evmv']
    xsbr1['qqH'] = qqXS[mass1]['XS_pb'] * br[mass1]['H_evmv']
    xsbr2['qqH'] = qqXS[mass2]['XS_pb'] * br[mass2]['H_evmv']
    xsbr ['qqH'] = qqXS[mass ]['XS_pb'] * br[mass ]['H_evmv']

print "Will interpolate %g from [%d, %d]" % (mass, mass1, mass2)

alpha = abs(mass2 - mass)/abs(mass2 - mass1) if mass1 != mass2 else 1.0; 
beta = 1 - alpha;
os.system("cp %s/%d/hww_%dj.input.root  %s/%g/hww_%dj%s.input.root" % (options.ddir,mass1,options.jets,options.ddir,mass,options.jets,options.postfix))
ofile  = ROOT.TFile( "%s/%g/hww_%dj%s.input.root" % (options.ddir,mass,options.jets,options.postfix) , "UPDATE" )

file1 = options.ddir+"/%d/hww_%dj_shape.txt" % (mass1, options.jets)
file2 = options.ddir+"/%d/hww_%dj_shape.txt" % (mass2, options.jets)
options.fileName = file1; options.mass = mass1;
DC1 = parseCard(open(file1,"r"), options)

options.fileName = file2; options.mass = mass2;
DC2 = parseCard(open(file2,"r"), options)

## Basic consistency check
if DC1.bins != DC2.bins: raise RuntimeError, "The two datacards have different bins: %s has %s, %s has %s" % (file1, DC1.bins, file2, DC2.bins)
if DC1.processes != DC2.processes: raise RuntimeError, "The two datacards have different processes: %s has %s, %s has %s" % (file1, DC1.processes, file2, DC2.processes)
if DC1.signals   != DC2.signals:   raise RuntimeError, "The two datacards have different signals: %s has %s, %s has %s" % (file1, DC1.signals, file2, DC2.signals)
if DC1.isSignal  != DC2.isSignal:  raise RuntimeError, "The two datacards have different isSignal: %s has %s, %s has %s" % (file1, DC1.isSignal, file2, DC2.isSignal)

if len(DC1.bins) != 1: raise RuntimeError, "This does not work on multi-channel"
obsline = [str(x) for x in DC1.obs.values()]; obskeyline = DC1.bins; cmax = 5;
keyline = []; expline = []; systlines = {}; systlines2 = {}
signals = []; backgrounds = []; shapeLines = []; 
paramSysts = {}; flatParamNuisances = {}
for (name,nf,pdf,args,errline) in DC1.systs:
    systlines[name] = [ pdf, args, errline, nf ]
for (name,nf,pdf,args,errline) in DC2.systs:
    systlines2[name] = [ pdf, args, errline, nf ]
for b,p,sig in DC1.keyline:
    if p not in DC2.exp[b].keys(): raise RuntimeError, "Process %s contributes to bin %s in card %s but not in card %s" % (p, b, file1, file2)
    rate = DC1.exp[b][p]
    if p in  ['ggH', 'qqH']:
        eff = rate/xsbr1[p]
        rate = eff * xsbr[p]
    if rate != 0: 
        histo = ofile.Get("histo_%s" % p);
        histo.Scale(rate/histo.Integral())
        ofile.WriteTObject(histo,"histo_%s" % p,"Overwrite");
    keyline.append( (b, p, DC1.isSignal[p]) )
    expline.append( "%.4f" % rate )
    if False:
        for name in systlines.keys():
            errline = systlines[name][2]
            if b in errline: 
                if p in errline[b]:
                    if options.log and pdf == "gmN" and errline[b][p] != 0: 
                        errline[b][p] = exp(alpha * log(systlines[name][2][b][p]) + beta*log(systlines2[name][2][b][p]))
                    else:
                        errline[b][p] = alpha * systlines[name][2][b][p] + beta*systlines2[name][2][b][p]
shapeLines.append( ("*",        obskeyline[0], [ "hww_%dj%s.input.root" % (options.jets,options.postfix),  "histo_$PROCESS" ]) )
shapeLines.append( ("data_obs", obskeyline[0], [ "hww_%dj%s.input.root" % (options.jets,options.postfix),  "histo_Data"     ]) )


xfile = open(options.ddir+"/%d/hww_%dj_shape%s.txt" % (mass, options.jets,options.postfix), "w")
xfile.write(" ".join(["imax %d number of bins" % len(DC1.bins)])+"\n")
xfile.write(" ".join(["jmax *  number of processes minus 1"])+"\n")
xfile.write(" ".join(["kmax *  number of nuisance parameters"])+"\n")
xfile.write(" ".join(["-" * 130])+"\n")
if shapeLines:
    chmax = max([max(len(p),len(c)) for p,c,x in shapeLines]);
    cfmt = "%-"+str(chmax)+"s ";
    for (process,channel,stuff) in shapeLines:
        xfile.write(" ".join(["shapes", cfmt % process, cfmt % channel, ' '.join(stuff)])+"\n")
    xfile.write(" ".join(["-" * 130])+"\n")

if obsline:
    cmax = max([cmax]+[len(l) for l in obskeyline]+[len(x) for x in obsline])
    cfmt = "%-"+str(cmax)+"s";
    xfile.write(" ".join(["bin         ", "  ".join([cfmt % x for x in obskeyline])])+"\n")
    xfile.write(" ".join(["observation ", "  ".join([cfmt % x for x in obsline])])+"\n")

xfile.write(" ".join(["-" * 150])+"\n")

pidline = []; signals = []; backgrounds = []
for (b,p,s) in keyline:
    if s:
        if p not in signals: signals.append(p)
        pidline.append(-len(DC1.signals)+signals.index(p)+1)
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
for (pname, pargs) in paramSysts.items():
    xfile.write(" ".join(["%-12s  param  %s" %  (pname, " ".join(pargs))])+"\n")

for pname in flatParamNuisances.iterkeys(): 
    xfile.write(" ".join(["%-12s  flatParam" % pname])+"\n")
