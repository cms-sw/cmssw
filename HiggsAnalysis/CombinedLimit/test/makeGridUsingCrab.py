#!/usr/bin/env python
from math import *
import os
from optparse import OptionParser
import ROOT
parser = OptionParser(usage="usage: %prog [options] workspace min max \nrun with --help to get list of options")
parser.add_option("-o", "--out",      dest="out",      default="TestGrid",  type="string", help="output file prefix")
parser.add_option("-O", "--options",  dest="options",  default="-M HybridNew --freq --testStat=Atlas --clsAcc=0",  type="string", help="options to use for combine")
parser.add_option("-n",  "--points",  dest="points",   default=10,  type="int",  help="Points to choose in the range")
parser.add_option("-T", "--toysH",      dest="T",        default=500, type="int",  help="Toys per point per iteration")
parser.add_option("-i", "--iterations", dest="i",        default=1, type="int",    help="Iterations per crab job")
parser.add_option("-I", "--interleave", dest="interl",   default=1, type="int",    help="If >1, excute only 1/I of the points in each job")
parser.add_option("-v", "--verbose",  dest="v",        default=0, type="int",    help="Verbosity")
parser.add_option("--fork",           dest="fork",     default=1,   type="int",  help="Cores to use")
parser.add_option("-s", "--seed",     dest="seed",     default=1, type="int",  help="Starting seed value (actual seed will be 10000 * point + seed; -1 = random)")
parser.add_option("-p", "--pretend",  dest="pretend",  default=False, action="store_true", help="Just print out the command, don't execute it")
(options, args) = parser.parse_args()
if len(args) != 3:
    parser.print_usage()
    exit(1)

workspace = args[0]
if workspace.endswith(".txt"):
    os.system("text2workspace.py -b %s -o %s.workspace.root" % (workspace, options.out))
    workspace = options.out+".workspace.root"
    print "Converted workspace to binary",workspace
    
min, max = float(args[1]), float(args[2])
dx = (max-min)/(options.points-1)

print "Creating executable script ",options.out+".sh"
script = open(options.out+".sh", "w")
script.write("""
#!/bin/bash
#############################################################
#
# Driver script for creating Hybrid or Frequentist grids
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
""")
for i in range(options.points):
    x = min + dx*i;
    seed = ("$((%d + $1))" % (i*10000)) if options.seed != -1 else "-1"
    interleave = "(( ($1 + %d) %% %d == 0 )) && " % (i, options.interl)
    script.write("{cond} ./combine {wsp} {opts} --fork {fork} -T {T} -i {i} --clsAcc 0 -v {v} -n {out} --saveHybridResult --saveToys -s {seed}  --singlePoint {x}\n".format(
                wsp=workspace, opts=options.options, fork=options.fork, T=options.T, seed=seed, out=options.out, x=x, v=options.v, i=options.i,
                cond=interleave
              ))

script.write("\n");
script.write("hadd %s.root higgsCombine*.root\n" % options.out)
script.write('echo "## Done at $(date)"\n');
script.close()
os.system("chmod +x %s.sh" % options.out)

if not os.path.exists("combine"):
    print "Creating a symlink to the combine binary"
    os.system("cp -s $(which combine) .")

print "Creating crab cfg ",options.out+".cfg"
cfg = open(options.out+".cfg", "w")
cfg.write("""
[CRAB]
jobtype = cmssw
scheduler = lsf
#scheduler = glite

[LSF]
queue = 1nh80

[CMSSW]
datasetpath = None
pset = None
output_file = {out}.root
total_number_of_events = 10
number_of_jobs = 10

[USER]
script_exe = {out}.sh
additional_input_files = combine,{wsp}
return_data = 1
""".format(wsp=workspace, out=options.out))
