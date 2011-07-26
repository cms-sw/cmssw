#!/usr/bin/env python
from math import *
import os
from optparse import OptionParser
import ROOT
parser = OptionParser(usage="usage: %prog [options] workspace(s) \nrun with --help to get list of options")
parser.add_option("-o", "--out",      dest="out",      default="TestGrid",  type="string", help="output file prefix")
parser.add_option("--lsf",            dest="lsf",      default=False,   action="store_true", help="Run on LSF instead of GRID (can be changed in .cfg file)")
parser.add_option("--condor",         dest="condor",   default=False, action="store_true", help="Run on condor_g instead of GRID (can be changed in .cfg file)")
parser.add_option("-q", "--queue",    dest="queue",    default="8nh",   type="string", help="LSF queue to use (can be changed in .cfg file)")
parser.add_option("-j", "--jobs",     dest="jobs",     default=10,      type="int",  help="Total number of jobs (can be changed in .cfg file)")
parser.add_option("-i", "--iteration",  dest="iters",  default=50000,   type="int", help="Number of iterations per chain")
parser.add_option("--tries",            dest="tries",  default=100,     type="int", help="Number of chains per worskpace among all job")
parser.add_option("-m", "--mass",     dest="mass",     default="120",   type="string", help="Mass, or comma-separated list of masses if using multiple workspaces")
parser.add_option("-O", "--options",  dest="options",  default="",      type="string", help="options to use for combine")
parser.add_option("-v", "--verbose",  dest="v",        default=0,       type="int",    help="Verbosity")
parser.add_option("-r", "--random",   dest="random",   default=False,   action="store_true", help="Use random seeds for the jobs")
parser.add_option("-P", "--priority", dest="prio",     default=False, action="store_true", help="Use PriorityUser role")
(options, args) = parser.parse_args()

workspaces = args
masses = options.mass.split(",")
if len(masses) == 1 and len(workspaces) != 1: masses = [ masses[0] for w in workspaces ]
elif len(workspaces) != len(masses): raise RuntimeError, "You must specify a number of masses equal to the number of workspaces, or just one mass for all of them"
for i, (w, m) in enumerate(zip(workspaces,masses)):
    if w.endswith(".txt"):
        os.system("text2workspace.py -b %s -o %s.workspace.root -m %s" % (w, options.out,m))
        workspaces[i] = options.out+".workspace.root"
        print "Converted workspace to binary",w

print "Creating executable script ",options.out+".sh"
script = open(options.out+".sh", "w")
script.write("""
#!/bin/bash
#############################################################
#
# Driver script for comuting Bayesian Limits 
#
# author: Giovanni Petrucciani, UCSD                       
#         from a similar script by Luca Lista, INFN        
#
##############################################################

i="$1"
if [ "$i" = "" ]; then
  echo "Error: missing job index"
  exit 1;
fi
echo "max events from CRAB: $MaxEvents"
n="$MaxEvents"
if [ "$n" = "" ]; then
  n="$2"
fi
if [ "$n" = "" ]; then
  echo "Error: missing number of experiments"
  exit 2;
fi

echo "## Starting at $(date)"
for I in $(seq 1 $n); do
""")
for i, (w, m) in enumerate(zip(workspaces,masses)):
    seed = ("$((%d + $i))" % (i*10000)) if options.random == False else "-1"
    script.write("./combine {wsp} -m {mass} -M MarkovChainMC {opts} -i {iters} --tries 1 -v {v} -n {out} -s {seed}\n".format(
                wsp=w, mass=m, opts=options.options, iters=options.iters, seed=seed, out=options.out, v=options.v
              ))
script.write("done\n\n");
script.write("hadd %s.root higgsCombine*.root\n" % options.out)
script.write('echo "## Done at $(date)"\n');
script.close()
os.system("chmod +x %s.sh" % options.out)

if not os.path.exists("combine"):
    print "Creating a symlink to the combine binary"
    os.system("cp -s $(which combine) .")

sched = "glite"
if options.lsf: sched = "lsf"
if options.condor: sched = "condor"

print "Creating crab cfg ",options.out+".cfg"
cfg = open(options.out+".cfg", "w")
cfg.write("""
[CRAB]
jobtype = cmssw
scheduler = {sched}

[LSF]
queue = {queue}

[CMSSW]
datasetpath = None
pset = None
output_file = {out}.root
total_number_of_events = {total}
number_of_jobs = {jobs}

[USER]
script_exe = {out}.sh
additional_input_files = combine,{wspall}
return_data = 1
""".format(wspall=(",".join(workspaces)), out=options.out, sched=sched, queue=options.queue, jobs=options.jobs, total=options.tries))

if options.prio: cfg.write("""
[GRID]
rb               = CERN
proxy_server     = myproxy.cern.ch
role             = priorityuser
retry_count      = 0
""")
