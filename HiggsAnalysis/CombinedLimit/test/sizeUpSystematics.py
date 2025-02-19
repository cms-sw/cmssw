#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit
from optparse import OptionParser
from pprint import pprint
from itertools import combinations
import os
import os.path
import json
from datetime import datetime, timedelta

class Timer:
    def __init__(self):
        self.times = list()
    def addLap(self, ev):
        dt = datetime.now()
        print "[Timer event at %s] %s" % (dt, ev)
        self.times.append( (ev, dt) )
    def printLaps(self):
        print "\nTotal running time: %s" % ( self.times[-1][1] - self.times[0][1] )
        print "Breakdown:"
        for i in range(len(self.times)-1):
            delta = ( self.times[i+1][1] - self.times[i][1] )
            print " [%s] to [%s]: %s sec." % (self.times[i][0], self.times[i+1][0], delta)

timer = Timer()

import os, errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST:
            pass
        else: raise


# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from HiggsAnalysis.CombinedLimit.DatacardParser import *

masses = list()
parser = OptionParser(usage="usage: %prog [options] datacard.txt [--dir output] \nrun with --help to get list of options")
parser.add_option("--depth", dest="depth",  default=1,  type="int", help="Scans excluding up to DEPTH and testing only up to DEPTH nuisances (0 means all in and all out if two-sided is specified).")
parser.add_option("--two-sided", dest="twosided",  default=False,  action='store_true', help="If specified, scans both including up to DEPTH and by removing down to DEPTH.")
parser.add_option("--dir", dest="dir",  default='systs',  type="string", help="Output directory for results and intermediates.")
parser.add_option("--masses", dest="masses",  default=[120],  type="string", action="callback", help="Which mass points to scan.",
                  callback = lambda option, opt_str, value, parser: masses.extend(map(lambda x: int(x), value.split(',')))
                  )
parser.add_option("--X-keep-global-nuisances", dest="keepGlobalNuisances", action="store_true", default=False, help="When excluding nuisances, do not exclude standard nuisances that are correlated CMS-wide even if their effect is small.")
addDatacardParserOptions(parser)
(options, args) = parser.parse_args()

if masses:
    options.masses = masses
    del masses

if len(args) == 0:
    parser.print_usage()
    exit(1)

options.fileName = args[0]
if options.fileName.endswith(".gz"):
    import gzip
    file = gzip.open(options.fileName, "rb")
    options.fileName = options.fileName[:-3]
else:
    file = open(options.fileName, "r")

DC = parseCard(file, options)

timer.addLap("Building structures")

# create output directory
mkdir_p(options.dir+"/log")
mkdir_p(options.dir+"/root")
OWD = os.getcwd()
os.chdir(options.dir) 


nuisances = map(lambda x: x[0], DC.systs )
if options.keepGlobalNuisances:
    nuisances = filter(lambda x: not globalNuisances.match(x), nuisances)

#pprint(nuisances)
g = open('nuisances.json','w')
g.write(json.dumps(nuisances, sort_keys=True, indent=2))
g.close()

# the factor two is related to the fact that we do singles and nMinusOnes
# (and pairs and nMinusTwos, etc) if twosided is specified
maxdepth = 2*options.depth if options.twosided else options.depth   
if len(nuisances) < maxdepth:
    raise RuntimeError, "Cannot process a depth that large."
    

# create combinations of interest
combinationsToRemove = list()
for level in xrange(options.depth+1):
    combinationsToRemove.extend( combinations(nuisances, level) )
    
    if options.twosided:
        if(level == len(nuisances)-level):
            # center of the pascal triangle, do only one
            continue
        combinationsToRemove.extend( combinations(nuisances, len(nuisances)-level) ) 

#pprint(combinationsToRemove)
g = open('combinationsToRemove.json','w')
g.write(json.dumps(combinationsToRemove, sort_keys=True, indent=2))
g.close()


# make hash strings for each combination to use in file names
# (yes, it is not human readable... it's life) 
import hashlib
import cPickle as pickle
combinationsHash = dict(
    [ (combination, hashlib.md5( pickle.dumps(combination) ).hexdigest() )
      for combination in combinationsToRemove
      ]
    )

#pprint(combinationsHash)
g = open('combinationsHash.json','w')
g.write(json.dumps(list(combinationsHash.items()), sort_keys=True, indent=2))
g.close()
           
    
# make list of jobs to run
jobs = dict() 
for combo in combinationsToRemove:
    nuisStr = '|'.join(combo)
    fName   = 'Syst.%s' % combinationsHash[combo]

    tempOut = "%s.root" % fName
    filterCmd = "text2workspace.py --verbose=2 %s -m %f  --X-exclude-nuisance='%s' -o root/%s &> log/%s.log" % (options.fileName, options.masses[0], nuisStr, tempOut, tempOut)

    if os.path.isfile('root/'+tempOut):
#        print "Not queuing text2workspace job for %s. (output already present)" % tempOut
        filterCmd = ''
        
    combineCmds = list()
    for mass in options.masses:
        combineMethod = "Asymptotic"
        combineOpts = "--minosAlgo=stepping -M %s -S 1 -m %g" % (combineMethod, mass)
        combineOut = "higgsCombine%s.%s.mH%g.root" % (fName, combineMethod, mass)

        combineCmd = "combine --verbose=1 root/%s %s -n %s &> log/%s.log && mv %s root/" % (tempOut, combineOpts, fName, combineOut, combineOut)

        if os.path.isfile('root/'+combineOut):
#            print "Not queuing combine job for %s. (output already present)" % combineOut
            combineCmd = ''

        combineCmds.append(combineCmd)

    jobs[combo] = (filterCmd, combineCmds, 'root/'+combineOut)

#pprint(jobs) 
g = open('jobs.json','w')
g.write(json.dumps(list(jobs.items()), sort_keys=True, indent=2))
g.close()


#function to be used in parallel 
def runCmd(cmd):
    if not cmd: return 0 
    from subprocess import call
    #print cmd
    return call(cmd, shell=True)

#make a pool of workers
from multiprocessing import Pool
pool = Pool()

timer.addLap("Running text2Workspace")
# run all the text2workspace in parallel
filterCmds = map(lambda x: x[0], jobs.values())
ret = pool.map(runCmd, filterCmds)
if reduce(lambda x, y: x+y, ret):
    raise RuntimeError, "Non-zero return code in text2workspace. Check logs."

timer.addLap("Running combine")
# run all the combine jobs in parallel
combineCmds = [ item for sublist in map(lambda x: x[1], jobs.values()) for item in sublist ] 
ret =  pool.map(runCmd, combineCmds)
if reduce(lambda x, y: x+y, ret):
    raise RuntimeError, "Non-zero return code in combine. Check logs."


timer.addLap("Harvesting root files")
# harvest limit values from the output files
limitsOut = 'limits.json'
if os.path.isfile(limitsOut):
#    print "Not harvesting root files. (output already present)"
    pass

else:
    limits = dict()
    for combination in combinationsToRemove:
        f = ROOT.TFile(jobs[combination][2])
        t = f.Get("limit")
    
        leaves = t.GetListOfLeaves()

        class Event(dict) :
            pass
        ev = Event()
        for i in range(0,leaves.GetEntries() ) :
            leaf = leaves.At(i)
            name = leaf.GetName()
            ev.__setattr__(name,leaf)

        valDict=dict()
        for iev in range(0,t.GetEntries()) :
            t.GetEntry(iev)
            valDict[int(1000*ev.quantileExpected.GetValue())] = ev.limit.GetValue()
        limits[combination] = valDict

    #pprint(limits)
    g = open(limitsOut,'w')
    g.write(json.dumps(list(limits.items()), sort_keys=True, indent=2))
    g.close()

os.chdir(OWD)

timer.addLap("Ranking results")
import rankSystematics

timer.addLap("All done")
timer.printLaps()
    

