#!/usr/bin/env python
from CMGTools.TTHAnalysis.plotter.mcAnalysis import *
import re, sys
systs = {}

from optparse import OptionParser
parser = OptionParser(usage="%prog [options] mc.txt cuts.txt systs.txt")
addMCAnalysisOptions(parser)
(options, args) = parser.parse_args()
options.weight = True
options.final  = True

mca  = MCAnalysis(args[0],options)
cuts = CutsFile(args[1],options)

from os.path import basename
binname = basename(args[1]).replace(".txt","")
report = mca.getYields(cuts)
procs = []; iproc = {}
for i,s in enumerate(mca.listSignals()):
    procs.append(s); iproc[s] = i-len(mca.listSignals())+1
for i,b in enumerate(mca.listBackgrounds()):
    procs.append(b); iproc[b] = i+1
allyields = mergeReports([y for x,y in report.iteritems()])
if 'data' not in report: 
    report['data'] = allyields

systs = {}
for sysfile in args[2:]:
    for line in open(sysfile, 'r'):
        if re.match("\s*#.*", line): continue
        line = re.sub("#.*","",line)
        field = [f.strip() for f in line.split(':')]
        (name, procmap, binmap, amount) = field
        if re.match(binmap,binname) == None: continue
        if name not in systs: systs[name] = []
        systs[name].append((re.compile(procmap),amount))
    sys.stderr.write("Loaded %d systematics\n" % len(systs))

for name in systs.keys():
    effmap = {}
    for p in procs:
        effect = "-"
        for (procmap,amount) in systs[name]:
            if re.match(procmap, p): effect = amount
        effmap[p] = effect
    systs[name] = effmap

print "## Datacard for cut file ",args[1]
print '##----------------------------------'
print 'bin         ',binname
print 'observation ',report['data'][-1][1][0]
print '##----------------------------------'
klen = max([7, len(binname)]+[len(p) for p in procs])
kpatt = " %%%ds "  % klen
fpatt = " %%%d.%df " % (klen,3)
print '##----------------------------------'
print 'bin         '," ".join([kpatt % binname  for p in procs])
print 'process     '," ".join([kpatt % p        for p in procs])
print 'process     '," ".join([kpatt % iproc[p] for p in procs])
print 'rate        '," ".join([fpatt % report[p][-1][1][0] for p in procs])
print '##----------------------------------'
for name,effmap in systs.iteritems():
    print '%-8s lnN' % name," ".join([kpatt % effmap[p]   for p in procs])
