#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *
from glob import glob
import os.path

MODULES = []

from CMGTools.TTHAnalysis.tools.btagSFs_POG import bTagSFEvent3WPErrs as btagSFEvent
#MODULES += [ ('btag', btagSFEvent) ]

from CMGTools.TTHAnalysis.tools.lepMVA_SF import AllLepSFs
MODULES += [ ('lep',AllLepSFs())  ]

from CMGTools.TTHAnalysis.tools.lepTrigger_SF import LepTriggerSF_Event
#MODULES += [ ('trig2l', LepTriggerSF_Event())  ]

from CMGTools.TTHAnalysis.tools.metLD_reshape import MetLDReshaper
#MODULES += [ ('metLD', MetLDReshaper()) ]

from CMGTools.TTHAnalysis.tools.btagRWTs_ND import BTag_RWT_EventErrs
#MODULES += [ ('btagRwt', BTag_RWT_EventErrs()) ]

class ScaleFactorProducer(Module):
    def __init__(self,name,booker,modules):
        Module.__init__(self,name,booker)
        self._modules = []
        self._xmodules = []
        for name,mod in modules:
            if hasattr(mod, 'listBranches'):
                self._xmodules.append(mod)
            else:
                self._modules.append((name,mod))
    def beginJob(self):
        self.t = PyTree(self.book("TTree","t","t"))
        #self.t.branch("SF_check_evt" ,"I")
        #self.t.branch("SF_check_lumi" ,"I")
        for name, mod in self._modules:
            self.t.branch("SF_%s" % name ,"F")
        for xmod in self._xmodules:
            for B in xmod.listBranches():
                self.t.branch("SF_%s" % B ,"F")
    def analyze(self,event):
        #self.t.SF_check_evt  = event.evt
        #self.t.SF_check_lumi = event.lumi
        for name, mod in self._modules:
            setattr(self.t, "SF_%s" % name, mod(event))
        for xmod in self._xmodules:
            keyvals = xmod(event)
            for B,V in keyvals.iteritems():
                setattr(self.t, "SF_%s" % B, V)
        self.t.fill()

import os, itertools

from optparse import OptionParser
parser = OptionParser(usage="%prog [options] <TREE_DIR> <OUT>")
parser.add_option("-d", "--dataset", dest="datasets",  type="string", default=[], action="append", help="Process only this dataset (or dataset if specified multiple times)");
parser.add_option("-c", "--chunk",   dest="chunks",    type="int",    default=[], action="append", help="Process only these chunks (works only if a single dataset is selected with -d)");
parser.add_option("-N", "--events",  dest="chunkSize", type="int",    default=500000, help="Default chunk size when splitting trees");
parser.add_option("-j", "--jobs",    dest="jobs",      type="int",    default=1, help="Use N threads");
parser.add_option("-p", "--pretend", dest="pretend",   action="store_true", default=False, help="Don't run anything");
parser.add_option("-q", "--queue",   dest="queue",     type="string", default=None, help="Run jobs on lxbatch instead of locally");
parser.add_option("-t", "--tree",    dest="tree",      default='ttHLepTreeProducerTTH', help="Pattern for tree name");
parser.add_option("-V", "--vector",  dest="vectorTree",action="store_true", default=True, help="Input tree is a vector")
(options, args) = parser.parse_args()

if len(args) != 2 or not os.path.isdir(args[0]) or not os.path.isdir(args[1]): 
    print "Usage: program <TREE_DIR> <OUT>"
    exit()
if len(options.chunks) != 0 and len(options.datasets) != 1:
    print "must specify a single dataset with -d if using -c to select chunks"
    exit()

jobs = []
for D in glob(args[0]+"/*"):
    fname = D+"/"+options.tree+"/"+options.tree+"_tree.root"
    if os.path.exists(fname):
        short = os.path.basename(D)
        if options.datasets != []:
            if short not in options.datasets: continue
        data = ("DoubleMu" in short or "MuEG" in short or "DoubleElectron" in short or "SingleMu" in short)
        if data: continue
        f = ROOT.TFile.Open(fname);
        #t = f.Get("ttHLepTreeProducerTTH" if options.vectorTree else "ttHLepTreeProducerBase")
        t = f.Get(options.tree)
        entries = t.GetEntries()
        f.Close()
        chunk = options.chunkSize
        if entries < chunk:
            print "  ",os.path.basename(D),("  DATA" if data else "  MC")," single chunk"
            jobs.append((short,fname,"%s/sfFriend_%s.root" % (args[1],short),data,xrange(entries),-1))
        else:
            nchunk = int(ceil(entries/float(chunk)))
            print "  ",os.path.basename(D),("  DATA" if data else "  MC")," %d chunks" % nchunk
            for i in xrange(nchunk):
                if options.chunks != []:
                    if i not in options.chunks: continue
                r = xrange(int(i*chunk),min(int((i+1)*chunk),entries))
                jobs.append((short,fname,"%s/sfFriend_%s.chunk%d.root" % (args[1],short,i),data,r,i))
print "\n"
print "I have %d taks to process" % len(jobs)

if options.queue:
    import os, sys
    basecmd = "bsub -q {queue} {dir}/lxbatch_runner.sh {dir} {cmssw} python {self} -N {chunkSize} -t {tree} {data} {output}".format(
                queue = options.queue, dir = os.getcwd(), cmssw = os.environ['CMSSW_BASE'], 
                self=sys.argv[0], chunkSize=options.chunkSize, tree=options.tree, data=args[0], output=args[1]
            )
    if options.vectorTree: basecmd += " --vector "
    # specify what to do
    for (name,fin,fout,data,range,chunk) in jobs:
        if chunk != -1:
            print "{base} -d {data} -c {chunk}".format(base=basecmd, data=name, chunk=chunk)
        else:
            print "{base} -d {data}".format(base=basecmd, data=name, chunk=chunk)
    exit()

maintimer = ROOT.TStopwatch()
def _runIt(myargs):
    (name,fin,fout,data,range,chunk) = myargs
    timer = ROOT.TStopwatch()
    fb = ROOT.TFile(fin)
    tb = fb.Get(options.tree)
    if options.vectorTree:
        tb.vectorTree = True
    else:
        tb.vectorTree = False
    nev = tb.GetEntries()
    if options.pretend:
        print "==== pretending to run %s (%d entries, %s) ====" % (name, nev, fout)
        return (name,(nev,0))
    print "==== %s starting (%d entries) ====" % (name, nev)
    booker = Booker(fout)
    el = EventLoop([ ScaleFactorProducer("sf",booker,MODULES), ])
    el.loop([tb], eventRange=range)
    booker.done()
    fb.Close()
    time = timer.RealTime()
    print "=== %s done (%d entries, %.0f s, %.0f e/s) ====" % ( name, nev, time,(nev/time) )
    return (name,(nev,time))

if options.jobs > 0:
    from multiprocessing import Pool
    pool = Pool(options.jobs)
    ret  = dict(pool.map(_runIt, jobs))
else:
    ret = dict(map(_runIt, jobs))
fulltime = maintimer.RealTime()
totev   = sum([ev   for (ev,time) in ret.itervalues()])
tottime = sum([time for (ev,time) in ret.itervalues()])
print "Done %d tasks in %.1f min (%d entries, %.1f min)" % (len(jobs),fulltime/60.,totev,tottime/60.)
