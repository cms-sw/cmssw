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
parser = OptionParser(usage="usage: %prog [options] [refmass1] mass3 > datacard3.txt \nrun with --help to get list of options")
parser.add_option("--xsbr",  dest="xsbr",   action="store_true", default=False, help="Use correct XS*BR for Higgs")
#parser.add_option("--doeff", dest="doeff",  action="store_true", default=False, help="Include interpolation of efficiency")
#parser.add_option("--nobg",  dest="nobg",   action="store_true", default=False, help="No interpolation of background normalizations")
parser.add_option("--log",   dest="log",    action="store_true", default=False, help="Use log-scale interpolation for yields (default is linear)")
parser.add_option("--ddir",  dest="ddir", type="string", default=".", help="Path to the datacards")
parser.add_option("--refmasses",  dest="refmasses", type="string",  default="hww.masses.txt", help="File containing the reference masses between which to interpolate (relative to --options.ddir)")
parser.add_option("--postfix",    dest="postfix",   type="string",  default="",               help="Postfix to add to datacard name")
parser.add_option("--extraThUncertainty",    dest="etu",   type="float",  default=0.0,               help="Add this amount linearly to gg->H cross section uncertainties")
(options, args) = parser.parse_args()
options.bin = True; options.stat = False
if len(args) not in [1,3]:
    parser.print_usage()
    exit(1)

from HiggsAnalysis.CombinedLimit.DatacardParser import *

refmasses = [ int(line) for line in open(options.ddir+"/"+options.refmasses,"r") ]

if len(args) == 1:
    mass = float(args[0])
    mass1 = refmasses[0]
    for m in refmasses[1:]:
        if abs(mass1 - mass) > abs(m - mass) or (abs(mass1 - mass) == abs(m - mass) and abs(mass1 - 164) < abs(m - 164)):
            mass1 = m
else:
    mass1 = int(args[0])
    mass  = float(args[1])

xsbr1 = { 'ggH':1.0, 'qqH':1.0 }
xsbr  = { 'ggH':1.0, 'qqH':1.0 }
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
sm4  = file2map(path+"SM4-600GeV.txt")
# create points at 450, 550 by interpolation
for M in (450,550):
    ggXS[M] = dict([ (key, 0.5*(ggXS[M+10][key] + ggXS[M-10][key])) for key in ggXS[M+10].iterkeys() ])
    qqXS[M] = dict([ (key, 0.5*(qqXS[M+10][key] + qqXS[M-10][key])) for key in qqXS[M+10].iterkeys() ])
    br[M] = dict([ (key, 0.5*(br[M+10][key] + br[M-10][key])) for key in br[M+10].iterkeys() ])
    sm4[M] = dict([ (key, 0.5*(sm4[M+10][key] + sm4[M-10][key])) for key in sm4[M+10].iterkeys() ])
if options.xsbr:
    xsbr1['ggH'] = ggXS[mass1]['XS_pb'] * br[mass1]['H_evmv'] 
    xsbr ['ggH'] = ggXS[mass ]['XS_pb'] * br[mass ]['H_evmv'] * sm4[mass ]['XS_over_SM'] * sm4[mass ]['brWW_over_SM']
    xsbr1['qqH'] = qqXS[mass1]['XS_pb'] * br[mass1]['H_evmv']
    xsbr ['qqH'] = qqXS[mass ]['XS_pb'] * br[mass ]['H_evmv'] * sm4[mass ]['brWW_over_SM']
else:
    xsbr['ggH'] = sm4[mass1]['XS_over_SM'] * sm4[mass1]['brWW_over_SM']
    xsbr['qqH'] = sm4[mass1]['brWW_over_SM']


print "Will interpolate %g from %d" % (mass, mass1)

for X in [ 'hwwof_0j_shape',  'hwwof_1j_shape',  'hwwsf_0j_shape',  'hwwsf_1j_shape', 'hww_2j_cut']:
    #print "Considering datacard ",X
    if "shape" in X:
        XS = X.replace("_shape","")
        os.system("cp %s/%d/%s.input.root  %s/%g/SM4_%s%s.input.root" % (options.ddir,mass1,XS,options.ddir,mass,XS,options.postfix))
        ofile  = ROOT.TFile( "%s/%g/SM4_%s%s.input.root" % (options.ddir,mass,XS,options.postfix) , "UPDATE" )

    file1 = options.ddir+"/%d/%s.txt" % (mass1, X)
    options.fileName = file1; options.mass = mass1;
    DC1 = parseCard(open(file1,"r"), options)

    if len(DC1.bins) != 1: raise RuntimeError, "This does not work on multi-channel"
    obsline = [str(x) for x in DC1.obs.values()]; obskeyline = DC1.bins; cmax = 5;
    keyline = []; expline = []; systlines = {}; 
    signals = []; backgrounds = []; shapeLines = []; 
    paramSysts = {}; flatParamNuisances = {}
    for (name,nf,pdf,args,errline) in DC1.systs:
        if options.etu != 0 and name in [ "QCDscale_ggH", "QCDscale_ggH1in", "QCDscale_ggH2in" ]:
            for b in errline.iterkeys(): 
                for p in errline[b].iterkeys():
                    if errline[b][p] != 0 and errline[b][p] != 1:
                        inflated = errline[b][p]+options.etu if errline[b][p] > 1 else errline[b][p]-options.etu
                        #print "Inflating uncertainty from %s to %s" % (errline[b][p], inflated);
                        errline[b][p] = inflated
        systlines[name] = [ pdf, args, errline, nf ]
    for b,p,sig in DC1.keyline:
        rate = DC1.exp[b][p]
        if p in  ['ggH', 'qqH']:
            eff = rate/xsbr1[p]
            rate = eff * xsbr[p]
        if (rate != 0) and ("shape" in X): 
            histo = ofile.Get("histo_%s" % p);
            histo.Scale(rate/histo.Integral())
            ofile.WriteTObject(histo,"histo_%s" % p,"Overwrite");
        keyline.append( (b, p, DC1.isSignal[p]) )
        expline.append( "%.4f" % rate )
    if "shape" in X:
        shapeLines.append( ("*",        obskeyline[0], [ "SM4_%s%s.input.root" % (XS,options.postfix),  "histo_$PROCESS" ]) )
        shapeLines.append( ("data_obs", obskeyline[0], [ "SM4_%s%s.input.root" % (XS,options.postfix),  "histo_Data"     ]) )


    xfile = open(options.ddir+"/%d/SM4_%s%s.txt" % (mass, X,options.postfix), "w")
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
