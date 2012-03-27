#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit
from optparse import OptionParser
from pprint import pprint
from itertools import combinations
from os import makedirs, chdir
import os.path
import json
from datetime import datetime, timedelta

times = list()
def addLap(ev):
    global times
    dt = datetime.now()
    print "%s: %s" % (ev, dt)
    times.append( (ev, dt) )

def printLaps():
    global times
    for i in range(len(times)-1):
        delta = ( times[i+1][1] - times[i][1] )
        print "From %s to %s: %s." % (times[i][0], times[i+1][0], delta)


addLap("Start")

# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from HiggsAnalysis.CombinedLimit.DatacardParser import *

masses = list()
parser = OptionParser(usage="usage: %prog [options] datacard.txt -o output \nrun with --help to get list of options")
parser.add_option("--depth", dest="depth",  default=1,  type="int", help="Scans excluding up to N and testing only up to N nuisances.")
parser.add_option("--masses", dest="masses",  default=[120],  type="string", action="callback", help="Which mass points to scan.",
                  callback = lambda option, opt_str, value, parser: masses.extend(map(lambda x: int(x), value.split(',')))
                  )
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

addLap("SystStart")

# create output directory
import errno
from os import makedirs, chdir
try:
    makedirs("systs")
except OSError as ex:
    if ex.errno == errno.EEXIST:
        pass
    else: raise

chdir("systs") 


nuisances = map(lambda x: x[0], DC.systs )
#pprint(nuisances)
g = open('nuisances.json','w')
g.write(json.dumps(nuisances, sort_keys=True, indent=2))
g.close()

# the factor two is related to the fact that we do singles and nMinusOnes
# (and pairs and nMinusTwos, etc)  
if len(nuisances) < 2*options.depth:
    raise RuntimeError, "Cannot process a depth that large."

# create combinations of interest
combinationsList = list()
for level in xrange(options.depth+1):
    combinationsList.extend( combinations(nuisances, level) )
    if(level == len(nuisances)-level):
        # center of the pascal triangle, do only one
        continue
    combinationsList.extend( combinations(nuisances, len(nuisances)-level) ) 

#pprint(combinationsList)
g = open('combinationsList.json','w')
g.write(json.dumps(combinationsList, sort_keys=True, indent=2))
g.close()


# make hash strings (yes, it is not human readable... it's life) 
import hashlib
import cPickle as pickle
combinationsHash = dict(
    [ (combination, hashlib.md5( pickle.dumps(combination) ).hexdigest() )
      for combination in combinationsList
      ]
    )

#pprint(combinationsHash)
g = open('combinationsHash.json','w')
g.write(json.dumps(list(combinationsHash.items()), sort_keys=True, indent=2))
g.close()
           
    
# make list of jobs to run
jobs = dict() 
for combo in combinationsList:
    nuisStr= '|'.join(combo)
    fName  = 'Syst.%s' % combinationsHash[combo]

    tempOut="%s.root" % fName
    filterCmd = "text2workspace.py %s --X-exclude-nuisance='%s' -o %s &> %s.log" % (options.fileName, nuisStr, tempOut, tempOut)

    if os.path.isfile(tempOut):
        print "Not queuing text2workspace for %s. (output already present)" % tempOut
        filterCmd = ''
        
    combineCmds = list()
    for mass in options.masses:
        combineMethod = "Asymptotic"
        combineOpts = "--minosAlgo=stepping -M %s -S 1 -m %g" % (combineMethod, mass)
        combineOut = "higgsCombine%s.%s.mH%g.root" % (fName, combineMethod, mass)

        combineCmd = "combine %s %s -n %s &> %s.log" % (tempOut, combineOpts, fName, combineOut)

        if os.path.isfile(combineOut):
            print "Not queuing combine for %s. (output already present)" % combineOut
            combineCmd = ''

        combineCmds.append(combineCmd)

    jobs[combo] = (filterCmd, combineCmds, combineOut)

#pprint(jobs) 
g = open('jobs.json','w')
g.write(json.dumps(list(jobs.items()), sort_keys=True, indent=2))
g.close()


#function to be used in parallel 
def runCmd(cmd):
    if not cmd: return 0 
    from subprocess import call
    print cmd
    return call(cmd, shell=True)

#make a pool of workers
from multiprocessing import Pool
pool = Pool()

addLap("Text2WorkspaceStart")
# run all the text2workspace in parallel
filterCmds = map(lambda x: x[0], jobs.values())
ret = pool.map(runCmd, filterCmds)
if reduce(lambda x, y: x+y, ret):
    raise RunTimeError, "Non-zero return code in text2workspace. Check logs."

addLap("CombineStart")
# run all the combine jobs in parallel
combineCmds = [ item for sublist in map(lambda x: x[1], jobs.values()) for item in sublist ] 
ret =  pool.map(runCmd, combineCmds)
if reduce(lambda x, y: x+y, ret):
    raise RunTimeError, "Non-zero return code in combine. Check logs."

addLap("ReductionStart")
# get the limit values from the output files
limits = dict()
for combo in combinationsList:
    f = ROOT.TFile(jobs[combo][2])
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

    limits[combo] = valDict   

#pprint(limits)
g = open('limits.json','w')
g.write(json.dumps(list(limits.items()), sort_keys=True, indent=2))
g.close()

chdir("..")

addLap("AllDone")

printLaps()
    

