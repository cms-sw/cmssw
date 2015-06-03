#!/usr/bin/env python
from CMGTools.MonoXAnalysis.plotter.mcAnalysis import *
import re, sys
systs = {}

from optparse import OptionParser
parser = OptionParser(usage="%prog [options] mc.txt cuts.txt systs.txt")
addMCAnalysisOptions(parser)
parser.add_option("-o",   "--out",    dest="outname", type="string", default=None, help="output name") 
parser.add_option("--od", "--outdir", dest="outdir", type="string", default=None, help="output name") 
parser.add_option("-v", "--verbose",  dest="verbose",  default=0,  type="int",    help="Verbosity level (0 = quiet, 1 = verbose, 2+ = more)")
parser.add_option("--asimov", dest="asimov", action="store_true", help="Asimov")
(options, args) = parser.parse_args()
options.weight = True
options.final  = True

mca  = MCAnalysis(args[0],options)
cuts = CutsFile(args[1],options)

from os.path import basename
binname = basename(args[1]).replace(".txt","") if options.outname == None else options.outname
outdir  = options.outdir+"/" if options.outdir else ""

report = mca.getYields(cuts)

procs = []; iproc = {}
signals, backgrounds = [], []
for i,s in enumerate(mca.listSignals()):
    signals.append(s)
    procs.append(s); iproc[s] = i-len(mca.listSignals())+1
for i,b in enumerate(mca.listBackgrounds()):
    backgrounds.append(b)
    procs.append(b); iproc[b] = i+1
allyields = mergeReports([y for x,y in report.iteritems()])

if options.asimov:
    report['data_obs'] = allyields
else:
    report['data_obs'] = report['data']

systs = {}
for sysfile in args[2:]:
    for line in open(sysfile, 'r'):
        if re.match("\s*#.*", line): continue
        line = re.sub("#.*","",line).strip()
        if len(line) == 0: continue
        field = [f.strip() for f in line.split(':')]
        if len(field) < 4:
            raise RuntimeError, "Malformed line %s in file %s"%(line.strip(),sysfile)
        elif len(field) == 4 or field[4] == "lnN":
            (name, procmap, binmap, amount) = field[:4]
            if re.match(binmap,binname) == None: continue
            if name not in systs: systs[name] = []
            systs[name].append((re.compile(procmap),amount))
        elif field[4] in ["envelop","shapeOnly","templates","alternateShapeOnly"]:
            if options.verbose > 0: print "Systematic %s of type %s not considered for counting analysis" %(field[0],field[4])
        else:
            raise RuntimeError, "Unknown systematic type %s" % field[4]
    if options.verbose > 0:
        print "Loaded %d systematics" % len(systs)
        print "Loaded %d envelop systematics" % len(systsEnv)


for name in systs.keys():
    effmap = {}
    for p in procs:
        effect = "-"
        for (procmap,amount) in systs[name]:
            if re.match(procmap, p): effect = amount
        effmap[p] = effect
    systs[name] = effmap

for signal in mca.listSignals():
    myout = outdir
    myout += "%s/" % signal
    myprocs = ( backgrounds + [ signal ] ) if signal in signals else backgrounds
    if not os.path.exists(myout): os.system("mkdir -p "+myout)
    datacard = open(myout+binname+".card.txt", "w");
    datacard = open(myout+binname+".card.txt", "w"); 
    datacard.write("## Datacard for cut file %s (signal %s)\n"%(args[1],signal))
    datacard.write("## Event selection: \n")
    for cutline in str(cuts).split("\n"):  datacard.write("##   %s\n" % cutline)
    if signal not in signals: datacard.write("## NOTE: no signal contribution found with this event selection.\n")
    datacard.write('##----------------------------------\n')
    datacard.write('bin         %s\n' % binname)
    klen = max([7, len(binname)]+[len(p) for p in procs])
    kpatt = " %%%ds "  % klen
    fpatt = " %%%d.%df " % (klen,3)
    datacard.write('observation'+("".join(fpatt % report['data_obs'][-1][1][0]))+"\n")
    datacard.write('##----------------------------------\n')
    datacard.write('##----------------------------------\n')
    datacard.write('bin             '+(" ".join([kpatt % binname     for p in myprocs]))+"\n")
    datacard.write('process         '+(" ".join([kpatt % p           for p in myprocs]))+"\n")
    datacard.write('process         '+(" ".join([kpatt % iproc[p]    for p in myprocs]))+"\n")
    datacard.write('rate            '+(" ".join([fpatt % report[p][-1][1][0] for p in myprocs]))+"\n")
    datacard.write('##----------------------------------\n')
    for name,effmap in systs.iteritems():
        datacard.write(('%-30s lnN' % name) + " ".join([kpatt % effmap[p]   for p in myprocs]) +"\n")
