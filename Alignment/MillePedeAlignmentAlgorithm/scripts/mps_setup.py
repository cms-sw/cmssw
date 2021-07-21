#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import os
import re
import sys
import shutil
import tarfile
import argparse
import subprocess
import Alignment.MillePedeAlignmentAlgorithm.mpslib.tools as mps_tools
import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib


parser = argparse.ArgumentParser(description = "Setup local mps database")
parser.add_argument("-m", "--setup-merge", dest = "setup_merge",
                    action = "store_true", default = False,
                    help = "setup pede merge job")
parser.add_argument("-a", "--append", action = "store_true", default = False,
                    help = "append jobs to existing list")
parser.add_argument("-M", "--memory", type = int, # seems to be obsolete
                    help = "memory (MB) to be allocated for pede")
parser.add_argument("-N", "--name", # remove restrictions on job name?
                    help = ("name to be assigned to the jobs; Whitespaces and "
                            "colons are not allowed"))
parser.add_argument("-w", "--weight", type = float,
                    help = "assign statistical weight")
parser.add_argument("-e", "--max-events", dest = "max_events", type = int,
                    help = "maximum number of events to process")

parser.add_argument("batch_script",
                    help = "path to the mille batch script template")
parser.add_argument("config_template",
                    help = "path to the config template")
parser.add_argument("input_file_list",
                    help = "path to the input file list")
parser.add_argument("n_jobs", type = int,
                    help = "number of jobs assigned to this dataset")
parser.add_argument("job_class",
                    help=("can be any of the normal LSF queues (8nm, 1nh, 8nh, "
                    "1nd, 2nd, 1nw, 2nw), special CAF queues (cmscaf1nh, "
                    "cmscaf1nd, cmscaf1nw) and special CAF pede queues "
                    "(cmscafspec1nh, cmscafspec1nd, cmscafspec1nw); if it "
                    "contains a ':' the part before ':' defines the class for "
                    "mille jobs and the part after defines the pede job class"))
parser.add_argument("job_name",
                    help = "name assigned to batch jobs")
parser.add_argument("merge_script",
                    help = "path to the pede batch script template")
parser.add_argument("mss_dir",
                    help = "name of the mass storage directory")

args = parser.parse_args(sys.argv[1:])


# setup mps database
lib = mpslib.jobdatabase()
lib.batchScript = args.batch_script
lib.cfgTemplate = args.config_template
lib.infiList = args.input_file_list
lib.nJobs = args.n_jobs
lib.classInf = args.job_class
lib.addFiles = args.job_name
lib.driver = "merge" if args.setup_merge else ""
lib.mergeScript = args.merge_script
lib.mssDirPool = ""
lib.mssDir = args.mss_dir
lib.pedeMem = args.memory


if not os.access(args.batch_script, os.R_OK):
    print("Bad 'batch_script' script name", args.batch_script)
    sys.exit(1)

if not os.access(args.config_template, os.R_OK):
    print("Bad 'config_template' file name", args.config_template)
    sys.exit(1)

if not os.access(args.input_file_list, os.R_OK):
    print("Bad input list file", args.input_file_list)
    sys.exit(1)

# ignore 'append' flag if mps database is not yet created
if not os.access("mps.db", os.R_OK): args.append = False

allowed_mille_classes = ("lxplus", "cmscaf1nh", "cmscaf1nd", "cmscaf1nw",
                         "cmscafspec1nh", "cmscafspec1nd", "cmscafspec1nw",
                         "8nm", "1nh", "8nh", "1nd", "2nd", "1nw", "2nw",
                         "cmsexpress","htcondor_cafalca_espresso","htcondor_espresso",
                         "htcondor_cafalca_microcentury","htcondor_microcentury",
                         "htcondor_cafalca_longlunch", "htcondor_longlunch",
                         "htcondor_cafalca_workday", "htcondor_workday",
                         "htcondor_cafalca_tomorrow", "htcondor_tomorrow",
                         "htcondor_cafalca_testmatch", "htcondor_testmatch",
                         "htcondor_cafalca_nextweek", "htcondor_nextweek")
if lib.get_class("mille") not in allowed_mille_classes:
    print("Bad job class for mille in class", args.job_class)
    print("Allowed classes:")
    for mille_class in allowed_mille_classes:
        print(" -", mille_class)
    sys.exit(1)

allowed_pede_classes = ("lxplus", "cmscaf1nh", "cmscaf1nd", "cmscaf1nw",
                        "cmscafspec1nh", "cmscafspec1nd", "cmscafspec1nw",
                        "8nm", "1nh", "8nh", "1nd", "2nd", "1nw", "2nw",
                        "htcondor_bigmem_espresso",
                        "htcondor_bigmem_microcentury",
                        "htcondor_bigmem_longlunch",
                        "htcondor_bigmem_workday",
                        "htcondor_bigmem_tomorrow",
                        "htcondor_bigmem_testmatch",
                        "htcondor_bigmem_nextweek")
if lib.get_class("pede") not in allowed_pede_classes:
    print("Bad job class for pede in class", args.job_class)
    print("Allowed classes:")
    for pede_class in allowed_pede_classes:
        print(" -", pede_class)
    sys.exit(1)

if args.setup_merge:
    if args.merge_script == "":
        args.merge_script = args.batch_script + "merge"
    if not os.access(args.merge_script, os.R_OK):
        print("Bad merge script file name", args.merge_script)
        sys.exit(1)

if args.mss_dir.strip() != "":
    if ":" in args.mss_dir:
        lib.mssDirPool = args.mss_dir.split(":")
        lib.mssDirPool, args.mss_dir = lib.mssDirPool[0], ":".join(lib.mssDirPool[1:])
        lib.mssDir = args.mss_dir

pedeMemMin = 1024 # Minimum memory allocated for pede: 1024MB=1GB

# Try to guess the memory requirements from the pede executable name.
# 2.5GB is used as default otherwise.
# AP - 23.03.2010
cms_process = mps_tools.get_process_object(args.config_template)
pedeMemDef = cms_process.AlignmentProducer.algoConfig.pedeSteerer.pedeCommand.value()
pedeMemDef = os.path.basename(pedeMemDef) # This is the pede executable (only the file name, eg "pede_4GB").
pedeMemDef = pedeMemDef.split("_")[-1]
pedeMemDef = pedeMemDef.replace("GB", "")
try:
    pedeMemDef = 1024*float(pedeMemDef)
    if pedeMemDef < pedeMemMin: pedeMemDef = pedeMemMin # pedeMemDef must be >= pedeMemMin.
except ValueError:
    pedeMemDef = int(1024*2.5)


# Allocate memory for the pede job.
# The value specified by the user (-M option) prevails on the one evinced from the executable name.
# AP - 23.03.2010
if not args.memory or args.memory < pedeMemMin:
    print("Memory request ({}) is < {}, using {}.".format(args.memory, pedeMemMin, pedeMemDef), end=' ')
    lib.pedeMem = args.memory = pedeMemDef

# Create the job directories
nJobExist = 0
if args.append and os.path.isdir("jobData"):
    # Append mode, and "jobData" exists
    jobs = os.listdir("jobData")
    job_regex = re.compile(r"job([0-9]{3})") # should we really restrict it to 3 digits?
    existing_jobs = [job_regex.search(item) for item in jobs]
    existing_jobs = [int(job.group(1)) for job in existing_jobs if job is not None]
    nJobExist = sorted(existing_jobs)[-1]

if nJobExist == 0 or nJobExist <=0 or nJobExist > 999: # quite rude method... -> enforce job number limit earlier?
    # Delete all
    mps_tools.remove_existing_object("jobData")
    os.makedirs("jobData")
    nJobExist = 0;

for j in range(1, args.n_jobs + 1):
    i = j+nJobExist
    jobdir = "job{0:03d}".format(i)
    print("jobdir", jobdir)
    os.makedirs(os.path.join("jobData", jobdir))

# build the absolute job directory path (needed by mps_script)
theJobData = os.path.abspath("jobData")
print("theJobData =", theJobData)

if args.append:
    # save current values
    tmpBatchScript = lib.batchScript
    tmpCfgTemplate = lib.cfgTemplate
    tmpInfiList    = lib.infiList
    tmpNJobs       = lib.nJobs
    tmpClass       = lib.classInf
    tmpMergeScript = lib.mergeScript
    tmpDriver      = lib.driver

    # Read DB file
    lib.read_db()

    # check if last job is a merge job
    if lib.JOBDIR[lib.nJobs] == "jobm":
        # remove the merge job
        lib.JOBDIR.pop()
        lib.JOBID.pop()
        lib.JOBSTATUS.pop()
        lib.JOBNTRY.pop()
        lib.JOBRUNTIME.pop()
        lib.JOBNEVT.pop()
        lib.JOBHOST.pop()
        lib.JOBINCR.pop()
        lib.JOBREMARK.pop()
        lib.JOBSP1.pop()
        lib.JOBSP2.pop()
        lib.JOBSP3.pop()

    # Restore variables
    lib.batchScript = tmpBatchScript
    lib.cfgTemplate = tmpCfgTemplate
    lib.infiList    = tmpInfiList
    lib.nJobs       = tmpNJobs
    lib.classInf    = tmpClass
    lib.mergeScript = tmpMergeScript
    lib.driver      = tmpDriver


# Create (update) the local database
for j in range(1, args.n_jobs + 1):
    i = j+nJobExist
    jobdir = "job{0:03d}".format(i)
    lib.JOBDIR.append(jobdir)
    lib.JOBID.append("")
    lib.JOBSTATUS.append("SETUP")
    lib.JOBNTRY.append(0)
    lib.JOBRUNTIME.append(0)
    lib.JOBNEVT.append(0)
    lib.JOBHOST.append("")
    lib.JOBINCR.append(0)
    lib.JOBREMARK.append("")
    lib.JOBSP1.append("")
    if args.weight is not None:
        lib.JOBSP2.append(str(args.weight))
    else:
        lib.JOBSP2.append("")
    lib.JOBSP3.append(args.name)

    # create the split card files
    cmd = ["mps_split.pl", args.input_file_list,
           str(j if args.max_events is None else 1),
           str(args.n_jobs if args.max_events is None else 1)]
    print(" ".join(cmd)+" > jobData/{}/theSplit".format(jobdir))
    with open("jobData/{}/theSplit".format(jobdir), "w") as f:
        try:
            subprocess.check_call(cmd, stdout = f)
        except subprocess.CalledProcessError:
            print("              split failed")
            lib.JOBSTATUS[i-1] = "FAIL"
    theIsn = "{0:03d}".format(i)

    # create the cfg file
    cmd = ["mps_splice.py", args.config_template,
           "jobData/{}/theSplit".format(jobdir),
           "jobData/{}/the.py".format(jobdir), theIsn]
    if args.max_events is not None:
        chunk_size = int(args.max_events/args.n_jobs)
        event_options = ["--skip-events", str(chunk_size*(j-1))]
        max_events = (args.max_events - (args.n_jobs-1)*chunk_size
                      if j == args.n_jobs    # last job gets the remaining events
                      else chunk_size)
        event_options.extend(["--max-events", str(max_events)])
        cmd.extend(event_options)
    print(" ".join(cmd))
    mps_tools.run_checked(cmd)

    # create the run script
    print("mps_script.pl {}  jobData/{}/theScript.sh {}/{} the.py jobData/{}/theSplit {} {} {}".format(args.batch_script, jobdir, theJobData, jobdir, jobdir, theIsn, args.mss_dir, lib.mssDirPool))
    mps_tools.run_checked(["mps_script.pl", args.batch_script,
                           "jobData/{}/theScript.sh".format(jobdir),
                           os.path.join(theJobData, jobdir), "the.py",
                           "jobData/{}/theSplit".format(jobdir), theIsn,
                           args.mss_dir, lib.mssDirPool])


# create the merge job entry. This is always done. Whether it is used depends on the "merge" option.
jobdir = "jobm";
lib.JOBDIR.append(jobdir)
lib.JOBID.append("")
lib.JOBSTATUS.append("SETUP")
lib.JOBNTRY.append(0)
lib.JOBRUNTIME.append(0)
lib.JOBNEVT.append(0)
lib.JOBHOST.append("")
lib.JOBINCR.append(0)
lib.JOBREMARK.append("")
lib.JOBSP1.append("")
lib.JOBSP2.append("")
lib.JOBSP3.append("")

lib.write_db();

# if merge mode, create the directory and set up contents
if args.setup_merge:
    shutil.rmtree("jobData/jobm", ignore_errors = True)
    os.makedirs("jobData/jobm")
    print("Create dir jobData/jobm")

    # We want to merge old and new jobs
    nJobsMerge = args.n_jobs+nJobExist

    # create  merge job cfg
    print("mps_merge.py -w {} jobData/jobm/alignment_merge.py {}/jobm {}".format(args.config_template, theJobData, nJobsMerge))
    mps_tools.run_checked(["mps_merge.py", "-w", args.config_template,
                           "jobData/jobm/alignment_merge.py",
                           os.path.join(theJobData, "jobm"), str(nJobsMerge)])

    # create merge job script
    print("mps_scriptm.pl {} jobData/jobm/theScript.sh {}/jobm alignment_merge.py {} {} {}".format(args.merge_script, theJobData, nJobsMerge, args.mss_dir, lib.mssDirPool))
    mps_tools.run_checked(["mps_scriptm.pl", args.merge_script,
                           "jobData/jobm/theScript.sh",
                           os.path.join(theJobData, "jobm"),
                           "alignment_merge.py", str(nJobsMerge), args.mss_dir,
                           lib.mssDirPool])


# Create a backup of batchScript, cfgTemplate, infiList (and mergeScript)
#   in jobData
backups = os.listdir("jobData")
bu_regex = re.compile(r"ScriptsAndCfg([0-9]{3})\.tar")
existing_backups = [bu_regex.search(item) for item in backups]
existing_backups = [int(bu.group(1)) for bu in existing_backups if bu is not None]
i = (0 if len(existing_backups) == 0 else sorted(existing_backups)[-1]) + 1
ScriptCfg = "ScriptsAndCfg{0:03d}".format(i)
ScriptCfg = os.path.join("jobData", ScriptCfg)
os.makedirs(ScriptCfg)
for f in (args.batch_script, args.config_template, args.input_file_list):
    shutil.copy2(f, ScriptCfg)
if args.setup_merge:
    shutil.copy2(args.merge_script, ScriptCfg)

with tarfile.open(ScriptCfg+".tar", "w") as tar: tar.add(ScriptCfg)
shutil.rmtree(ScriptCfg)


# Write to DB
lib.write_db();
lib.read_db();
lib.print_memdb();
