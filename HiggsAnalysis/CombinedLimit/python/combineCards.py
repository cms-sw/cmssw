import re
from sys import argv
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-s", "--stat",   dest="stat",    default=False, action="store_true")  # ignore systematic uncertainties to consider statistical uncertainties only
parser.add_option("-a", "--asimov", dest="asimov",  default=False, action="store_true")
(options, args) = parser.parse_args()

from HiggsAnalysis.CombinedLimit.DatacardParser import *

obsline = []; obskeyline = [] ;
keyline = []; expline = []; systlines = {}
signals = []; backgrounds = []; 
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
        bout = label if singlebin else label+bin
        obskeyline.append(bout)
        for (p,e) in DC.exp[b].items(): # so that we get only self.DC.processes contributing to this bin
            expline.append("%.4f" % e)
            keyline.append((bout, p, DC.isSignal[p]))
    # systematics
    for (lsyst,pdf,pdfargs,errline) in DC.systs:
        systeffect = {}
        for b in DC.bins:
            bout = label if singlebin else label+bin
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
print "kmax %d number of nuisance parameters" % len(systlines)
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

for name,(pdf,pdfargs,effect) in systlines.items():
    systline = []
    for b,p,s in keyline:
        try:
            systline.append(effect[b][p])
        except KeyError:
            systline.append("-");
    print hfmt % ("%-12s   %s  %s" % (name, pdf, " ".join(pdfargs))), "  ".join([cfmt % x for x in systline])
