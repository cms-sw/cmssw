#!/usr/bin/env python
import re
from sys import argv
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-s", "--stat",   dest="stat",    default=False, action="store_true")  # ignore systematic uncertainties to consider statistical uncertainties only
parser.add_option("-a", "--asimov", dest="asimov",  default=False, action="store_true")
(options, args) = parser.parse_args()
options.bin = True # fake that is a binary output, so that we parse shape lines

from HiggsAnalysis.CombinedLimit.DatacardParser import *

obsline = []; obskeyline = [] ;
keyline = []; expline = []; systlines = {}
signals = []; backgrounds = []; shapeLines = []
paramSysts = {}
cmax = 5 # column width
for ich,fname in enumerate(args):
    label = "ch%d" % (ich+1)
    if "=" in fname: (label,fname) = fname.split("=")
    file = open(fname, "r")
    DC = parseCard(file, options)
    singlebin = (len(DC.bins) == 1)
    if not singlebin: label += "_";
    # expectations
    for b in DC.bins:
        bout = label if singlebin else label+b
        obskeyline.append(bout)
        for (p,e) in DC.exp[b].items(): # so that we get only self.DC.processes contributing to this bin
            expline.append("%.4f" % e)
            keyline.append((bout, p, DC.isSignal[p]))
    # systematics
    for (lsyst,pdf,pdfargs,errline) in DC.systs:
        systeffect = {}
        if pdf == "param":
            if paramSysts.has_key(lsyst):
               if paramSysts[lsyst] != pdfargs: raise RuntimeError, "Parameter uncerainty %s mismatch between cards." % lsyst
            else:
                paramSysts[lsyst] = pdfargs
            continue
        for b in DC.bins:
            bout = label if singlebin else label+b
            if not systeffect.has_key(bout): systeffect[bout] = {} 
            for p in DC.exp[b].keys(): # so that we get only self.DC.processes contributing to this bin
                r = "%.3f" % errline[b][p] if pdf != "gmN" else str(errline[b][p])
                if errline[b][p] == 0: r = "-"
                if len(r) > cmax: cmax = len(r) # get max col length, as it's more tricky do do it later with a map
                systeffect[bout][p] = r
        if systlines.has_key(lsyst):
            (otherpdf, otherargs, othereffect) = systlines[lsyst]
            if otherpdf != pdf: 
                raise RuntimeError, "File %s defines systematic %s as using pdf %s, while a previous file defines it as using %s" % (fname,lsyst,pdf,otherpdf)
            if pdf == "gmN" and int(pdfargs[0]) != int(otherargs[0]): 
                raise RuntimeError, "File %s defines systematic %s as using gamma with %s events in sideband, while a previous file has %s" % (fname,lsyst,pdfargs[0],otherargs[0])
            for b,v in systeffect.items(): othereffect[b] = v;
        else:
            pdfargs = [ str(x) for x in pdfargs ]
            systlines[lsyst] = (pdf,pdfargs,systeffect)
    # put shapes, if available
    if len(DC.shapeMap):
        for b in DC.bins:
            bout = label if singlebin else label+b
            p2sMap  = DC.shapeMap[b]   if DC.shapeMap.has_key(b)   else {}
            p2sMapD = DC.shapeMap['*'] if DC.shapeMap.has_key('*') else {}
            for p, x in p2sMap.items():
                xrep = [xi.replace("$CHANNEL",bout) for xi in x]
                shapeLines.append((p,bout,xrep))
            for p, x in p2sMapD.items():
                if p2sMap.has_key(p): continue
                xrep = [xi.replace("$CHANNEL",bout) for xi in x]
                shapeLines.append((p,bout,xrep))
    # combine observations, but remove line if any of the datacards doesn't have it
    if len(DC.obs) == 0:
        obsline = None
    elif obsline != None:
        obsline += [str(DC.obs[b]) for b in DC.bins]; 

bins = []
for (b,p,s) in keyline:
    if b not in bins: bins.append(b)
    if s:
        if p not in signals: signals.append(p)
    else:
        if p not in backgrounds: backgrounds.append(p)

print "Combination of", ", ".join(args)
print "imax %d number of bins" % len(bins)
print "jmax %d number of processes minus 1" % (len(signals) + len(backgrounds) - 1)
print "kmax %d number of nuisance parameters" % (len(systlines) + len(paramSysts))
print "-" * 130

if shapeLines:
    chmax = max([max(len(p),len(c)) for p,c,x in shapeLines]);
    cfmt = "%-"+str(chmax)+"s ";
    for (process,channel,stuff) in shapeLines:
        print "shapes", cfmt % process, cfmt % channel, ' '.join(stuff);
    print "-" * 130

if obsline:
    cmax = max([cmax]+[len(l) for l in obskeyline]+[len(x) for x in obsline])
    cfmt = "%-"+str(cmax)+"s";
    print "bin         ", "  ".join([cfmt % x for x in obskeyline])
    print "observation ", "  ".join([cfmt % x for x in obsline])

print "-" * 130

pidline = []; signals = []; backgrounds = []
for (b,p,s) in keyline:
    if s:
        if p not in signals: signals.append(p)
        pidline.append(-signals.index(p))
    else:
        if p not in backgrounds: backgrounds.append(p)
        pidline.append(1+backgrounds.index(p))
cmax = max([cmax]+[max(len(p),len(b)) for p,b,s in keyline]+[len(e) for e in expline])
hmax = max([10] + [len("%-12s  %s %s" % (l,p,a)) for l,(p,a,e) in systlines.items()])
cfmt  = "%-"+str(cmax)+"s"; hfmt = "%-"+str(hmax)+"s  ";
print hfmt % "bin",     "  ".join([cfmt % p for p,b,s in keyline])
print hfmt % "process", "  ".join([cfmt % b for p,b,s in keyline])
print hfmt % "process", "  ".join([cfmt % x for x in pidline])
print hfmt % "rate",    "  ".join([cfmt % x for x in expline])

print "-" * 130

sysnamesSorted = systlines.keys(); sysnamesSorted.sort()
for name in sysnamesSorted:
    (pdf,pdfargs,effect) = systlines[name]
    systline = []
    for b,p,s in keyline:
        try:
            systline.append(effect[b][p])
        except KeyError:
            systline.append("-");
    print hfmt % ("%-12s   %s  %s" % (name, pdf, " ".join(pdfargs))), "  ".join([cfmt % x for x in systline])
for (pname, pargs) in paramSysts.items():
    print "%-12s  param  %s" %  (pname, " ".join(pargs))
