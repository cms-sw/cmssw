#!/usr/bin/env python
from math import *
import os,sys
from optparse import OptionParser
import ROOT
parser = OptionParser()#usage="usage: %prog [options] workspace min max \nrun with --help to get list of options")
parser.add_option("-o", "--out",      dest="out",      default="TestGrid",  type="string", help="output file prefix")
parser.add_option("--lsf",            dest="lsf",      default=False, action="store_true", help="Run on LSF instead of GRID (can be changed in .cfg file)")
parser.add_option("--condor",         dest="condor",   default=False, action="store_true", help="Run on condor_g instead of GRID (can be changed in .cfg file)")
parser.add_option("--glidein",        dest="glide",    default=False, action="store_true", help="Use glide-in scheduler instead of glite")
parser.add_option("--server",         dest="server",   default=False, action="store_true", help="Use crab server")
parser.add_option("-q", "--queue",    dest="queue",    default="8nh",   type="string", help="LSF queue to use (can be changed in .cfg file)")
parser.add_option("-O", "--options",  dest="options",  default="--freq ",  type="string", help="options to use for combine")
parser.add_option("-n", "--points",   dest="points",   default=(10,10),     nargs=2, type="int",  help="Points to choose in the range (note: both endpoints are included)")
parser.add_option("", "--parameterRanges",   dest="parameterRanges", nargs=4, type="float",  help="physics model parameter ranges (should be x1,x2,y1,y2) used to induce boundaries for physical regions. Default is range of POI points")
parser.add_option("-P", "--POI",      dest="POI",      default=("r","MH"),  nargs=2, type="string",  help="Parameters")
parser.add_option("-T", "--toysH",    dest="T",        default=50, type="int",  help="Toys per point per iteration")
parser.add_option("-t", "--toys",     dest="t",        default=10,  type="int",  help="Total number of iterations per point among all jobs (can be changed in .cfg file)")
parser.add_option("-j", "--jobs",     dest="j",        default=10,  type="int",  help="Total number of jobs (can be changed in .cfg file)")
parser.add_option("-I", "--interleave", dest="interl", default=1, type="int",    help="If >1, excute only 1/I of the points in each job")
parser.add_option("-v", "--verbose",  dest="v",        default=0, type="int",    help="Verbosity")
parser.add_option("-r", "--random",   dest="random",   default=True, action="store_true", help="Use random seeds for the jobs")
parser.add_option("--priority", dest="prio",     default=False, action="store_true", help="Use PriorityUser role")
parser.add_option("-u", "--uidir",    dest="uidir", default="", help="Sepcify a CRAB UI directory.  If not provided using the normal crab_0_timestamp")
parser.add_option("-m", "--mass",     dest="mass",	default=120, type="float",help="Hypothesis mass (mH)")
#parser.add_option("--fork",           dest="fork",     default=1,   type="int",  help="Cores to use (leave to 1)") # no fork in batch jobs for now
(options, args) = parser.parse_args()
if len(args) != 5:
    parser.print_usage()
    exit(1)
options.fork = 1 ## NEVER EVER FORK IN GRID JOBS. NOT ALLOWED BY THE SYSTEM ##

workspace = args[0]
if workspace.endswith(".txt"):
    os.system("text2workspace.py -b %s -o %s.workspace.root -m %f" % (workspace, options.out, options.mass))
    workspace = options.out+".workspace.root"
    print "Converted workspace to binary",workspace

#if options.diagnosticRun : options.points = 1 

xmin, xmax = float(args[1]), float(args[2])
dx = (xmax-xmin)/(options.points[0]-1) if options.points[0] > 1 else 0
ymin, ymax = float(args[3]), float(args[4])
dy = (ymax-ymin)/(options.points[1]-1) if options.points[1] > 1 else 0
points = []
for x in [ xmin + dx*i for i in range(options.points[0]) ]:
    for y in [ ymin + dy*i for i in range(options.points[1]) ]:
        points.append((x,y))

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

## Save memory on batch systems by avoinding a redundant fork when only one child will be ever spawned
nchild={fork};
if  [[ "$nchild" == "1" && "$n" == "1" ]]; then
    nchild=0;
fi;
echo "## Starting at $(date)"
""".format(fork=options.fork))
for i,x in enumerate(points):
    seed = ("$((%d + $i))" % (i*10000)) if options.random == False else "-1"
    interleave = "(( ($i + %d) %% %d == 0 )) && " % (i, options.interl)
    if ( options.parameterRanges ) :
		xrngemin = options.parameterRanges[0]
		xrngemax = options.parameterRanges[1]
		yrngemin = options.parameterRanges[2]
		yrngemax = options.parameterRanges[3]
    else : 
		xrngemin = xmin
		xrngemax = xmax
		yrngemin = ymin
		yrngemax = ymax

    ranges = "--setPhysicsModelParameterRanges %s=%g,%g:%s=%g,%g" % (options.POI[0], xrngemin,xrngemax,options.POI[1],yrngemin,yrngemax)  # Needed for naming of HypoTestResults and physics boundaries
    toys = "$n"
    what = "--singlePoint %s=%g,%s=%g " % (options.POI[0], x[0], options.POI[1], x[1]) ;
    script.write("{cond} ./combine {wsp} -M HybridNew {opts} -m {mass} --testStat=PL --rule=CLsplusb --fork $nchild -T {T} --clsAcc 0 -v {v} -n {out} --saveHybridResult --saveToys -s {seed} -i {toys} {what}  {ranges} \n".format(
                wsp=workspace, opts=options.options, fork=options.fork, T=options.T, seed=seed, out=options.out, what=what, v=options.v,
                cond=interleave, toys=toys,mass=options.mass,ranges=ranges
              ))

script.write("\n");
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
if options.glide: sched = "glidein"

uidir_line = ''
if options.uidir:
    uidir_line = "ui_working_dir = %s" % options.uidir

print "Creating crab cfg ",options.out+".cfg"
cfg = open(options.out+".cfg", "w")
cfg.write("""
[CRAB]
jobtype = cmssw
scheduler = {sched}
use_server = {srv}

[LSF]
queue = {queue}

[CMSSW]
datasetpath = None
pset = None
output_file = {out}
total_number_of_events = {total}
number_of_jobs = {jobs}

[USER]
script_exe = {scriptout}.sh
additional_input_files = combine,{wsp}
return_data = 1
{uidir_line}
""".format(
    wsp=workspace, out=options.out+'.root',
    scriptout=options.out,
    sched=sched, srv=(1 if options.server else 0),
    queue=options.queue, jobs=options.j, total=options.t,
    uidir_line = uidir_line
))


if options.prio: cfg.write("""
[GRID]
rb               = CERN
proxy_server     = myproxy.cern.ch
role             = priorityuser
retry_count      = 0
""")
