import re
from sys import argv
from optparse import OptionParser

args = argv[1:]
obsline = []; obskeyline = [] ;
keyline = []; expline = []; systlines = {}
signals = []; backgrounds = []; 
cmax = 5 # column width
for ich,fname in enumerate(args):
    label = "ch%d" % (ich+1)
    if "=" in fname: (label,fname) = fname.split("=")
    label += "_";
    file = open(fname, "r")
    myobsline = None; myobskeyline = []
    mykeyline = []; myexpline = []; mysystlines = {}
    processline = []; sigline = []; binline  = []; 
    singlebin = False
    for l in file:
        f = l.split();
        if len(f) < 1: continue
        if f[0] == "imax" and f[1] == "1": 
            singlebin = True; label = label[:-1] # remove trailing "_" from label
        if f[0] == "Observation" or f[0] == "observation": 
            myobsline = f[1:]
            binline = [] # reset in case bin names had been declared twice
        if f[0] == "bin": 
            binline = f[1:] # binline before processes
        if f[0] == "process": 
            if processline == []: # first line contains names
                processline = f[1:]
                if len(binline) != len(processline): raise RuntimeError, "Error in file %s: 'bin' line has a different length than 'process' line." % fname
                continue
            sigline = f[1:] # second line contains ids
            if len(sigline) != len(processline): raise RuntimeError, "Error in file %s: 'bin' line has a different length than 'process' line." % fname
            for i,b in enumerate(binline):
                p = processline[i];
                s = (int(sigline[i]) <= 0) # <=0 for signals, >0 for backgrounds
                binkey = label if singlebin else label+b
                mykeyline.append((binkey, processline[i], s))
                if binkey not in myobskeyline: myobskeyline.append(binkey)
            keyline += mykeyline
        if f[0] == "rate":
            if len(f[1:]) != len(mykeyline): raise RuntimeError, "Error in file %s: rate line has length %d, while bins and process lines have length %d" % (fname, len(f[1:]), len(mykeyline))
            expline += f[1:]
            break;    
    # systematics
    for l in file:
        if l.startswith("--"): continue
        l = re.sub("\\s-+(\\s|$)"," 0\\1",l); # change - or -- into a 0, preserving spaces
        f = l.split();
        lsyst = f[0]; pdf = f[1]; pdfargs = []; numbers = f[2:];
        if pdf == "lnN":
            #convert 1.0 in 0.0
            for i,x in enumerate(numbers):
                if float(x) == 1: numbers[i] = "0"
        elif pdf == "gmM":
            pass # nothing special to do
        elif pdf == "gmN":
            pdfargs = [int(f[2])]; numbers = f[3:];
        else:
            raise RuntimeError, "Unsupported pdf %s" % pdf
        if len(numbers) < len(mykeyline): raise RuntimeError, "Malformed systematics line %s of length %d: while bins and process lines have length %d" % (lsyst, len(numbers), len(mykeyline))
        systeffect = {}
        for (b,p,s),r in zip(mykeyline,numbers[0:len(keyline)]):
            if not systeffect.has_key(b): systeffect[b] = {} # no label+b since mykeyline has it already
            systeffect[b][p] = r
            if len(r) > cmax: cmax = len(r) # fix this as it's more tricky do do it with a map
        if systlines.has_key(lsyst):
            (otherpdf, otherargs, othereffect) = systlines[lsyst]
            if otherpdf != pdf: raise RuntimeError, "File %s defines systematic %s as using pdf %s, while a previous file defines it as using %s" % (fname,lsyst,pdf,otherpdf)
            if pdf == "gmN" and pdfargs[0] != otherargs[0]: raise RuntimeError, "File %s defines systematic %s as using gamma with %d events in sideband, while a previous file has %d" % (fname,lsyst,pdfargs[0],otherargs[0])
            for b,v in systeffect.items(): othereffect[b] = v;
        else:
            systlines[lsyst] = (pdf,pdfargs,systeffect)
    # combine observations, but remove line if any of the datacards doesn't have it
    if myobsline == None:
        obsline = None
    elif obsline != None:
        obsline += myobsline; obskeyline += myobskeyline

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
    print "Observation ", "  ".join([cfmt % x for x in obsline])

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
hmax = max([10] + [len("%s  %s %s" % (l,p,a)) for l,(p,a,e) in systlines.items()])
cfmt  = "%-"+str(cmax)+"s"; hfmt = "%-"+str(hmax)+"s  ";
print hfmt % "bin",     "  ".join([cfmt % p for p,b,s in keyline])
print hfmt % "process", "  ".join([cfmt % b for p,b,s in keyline])
print hfmt % "process", "  ".join([cfmt % x for x in pidline])
print hfmt % "rate",    "  ".join([cfmt % x for x in expline])

print "-" * 130

for name,(pdf,pdfargs,effect) in systlines.items():
    systline = []
    for b,p,s in keyline:
        eff = "--"
        try:
            systline.append(effect[b][p] if effect[b][p] != "0" else "--")
        except KeyError:
            systline.append("--");
    print hfmt % ("%s   %s  %s" % (name, pdf, " ".join(pdfargs))), "  ".join([cfmt % x for x in systline])
