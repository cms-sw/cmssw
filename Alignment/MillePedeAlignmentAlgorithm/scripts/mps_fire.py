#!/usr/bin/env python3
#  Submit jobs that are setup in local mps database to batch system
#
#  The bsub sytax: bsub -J 'jobname' -q 'queue name' theProgram
#  The jobname will be something like MP_2015.
#  The queue name is derived from lib.classInfo.
#  The program is theScrip.sh located in each job-directory.
#  There may be the other option -R (see man bsub for info).
#
#  Usage:
#
#  mps_fire.py [-a] [-m [-f]] [maxjobs]
#  mps_fire.py -h

from __future__ import print_function
from builtins import range
import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib
import Alignment.MillePedeAlignmentAlgorithm.mpslib.tools as mps_tools
import os
import sys
import glob
import shutil
if sys.version_info[0]>2:
  import _pickle as cPickle
else:
  import cPickle
import subprocess
import re
import argparse

def forward_proxy(rundir):
    """Forward proxy to location visible from the batch system.

    Arguments:
    - `rundir`: directory for storing the forwarded proxy
    """

    if not mps_tools.check_proxy():
        print("Please create proxy via 'voms-proxy-init -voms cms -rfc'.")
        sys.exit(1)

    local_proxy = subprocess.check_output(["voms-proxy-info", "--path"]).decode().strip()
    shutil.copyfile(local_proxy, os.path.join(rundir,".user_proxy"))


def write_HTCondor_submit_file_pede(path, script, config, lib):
    """Writes 'job.submit' file in `path`.

    Arguments:
    - `path`: job directory
    - `script`: script to be executed
    - `config`: cfg file
    - `lib`: MPS lib object
    """

    resources = lib.get_class("pede").split("_")[1:] # strip off 'htcondor'
    job_flavour = resources[-1]

    job_submit_template="""\
universe              = vanilla
executable            = {script:s}
output                = {jobm:s}/STDOUT
error                 = {jobm:s}/STDOUT
log                   = {jobm:s}/HTCJOB
notification          = Always
transfer_output_files = ""
request_memory        = {pedeMem:d}M

# adapted to space used on eos for binaries:
request_disk          = {disk:d}

# adapted to threads parameter in pede options and number of available cores
request_cpus          = {cpus:d}

+JobFlavour           = "{flavour:s}"
"""
    if "bigmem" in resources:
        job_submit_template += """\
+BigMemJob            = True
+AccountingGroup      = "group_u_CMS.e_cms_caf_bigmem"

# automatically remove the job if the submitter has no permissions to run a BigMemJob
periodic_remove       = !regexp("group_u_CMS.e_cms_caf_bigmem", AccountingGroup) && BigMemJob =?= True
"""
    job_submit_template += "\nqueue\n"

    print("Determine number of pede threads...")
    cms_process = mps_tools.get_process_object(os.path.join(Path, mergeCfg))
    pede_options = cms_process.AlignmentProducer.algoConfig.pedeSteerer.options.value()
    n_threads = 1
    for option in pede_options:
        if "threads" in option:
            n_threads = option.replace("threads", "").strip()
            n_threads = max(map(lambda x: int(x), n_threads.split()))
            break
    if n_threads > 16: n_threads = 16 # HTCondor machines have (currently) 16
                                      # cores, i.e. we ensure here that the job
                                      # would fit core-wise on one machine

    print("Determine required disk space on remote host...")
    # determine usage by each file instead of whole directory as this is what
    # matters for the specified disk usage:
    spco = subprocess.check_output # to make code below more less verbose
    opj = os.path.join             # dito
    cmd = ["du", "--apparent-size"]
    disk_usage = [int(item.split()[0])
                  for directory in ("binaries", "monitors", "tree_files")
                  for item
                  in spco(cmd+
                          glob.glob(opj(lib.mssDir, directory, "*"))).decode().splitlines()]
    disk_usage = sum(disk_usage)
    disk_usage *= 1.1 # reserve 10% additional space

    job_submit_file = os.path.join(Path, "job.submit")
    with open(job_submit_file, "w") as f:
        f.write(job_submit_template.format(script = os.path.abspath(script),
                                           jobm = os.path.abspath(path),
                                           pedeMem = lib.pedeMem,
                                           disk = int(disk_usage),
                                           cpus = n_threads,
                                           flavour = job_flavour))

    return job_submit_file

def write_HTCondor_submit_file_mille(path, script, lib, proxy_path=None):
    """Writes 'job.submit' file in `path`.

    Arguments:
    - `path`: job directory
    - `script`: script to be executed
    - `lib`: MPS lib object
    - `proxy_path`: path to proxy (only used in case of requested proxy forward)
    """

    resources = lib.get_class("mille").split("_")[1:] # strip off 'htcondor'
    job_flavour = resources[-1]

    job_submit_template="""\
universe              = vanilla
executable            = {script:s}
output                = {jobm:s}/STDOUT
error                 = {jobm:s}/STDOUT
log                   = {jobm:s}/HTCJOB
notification          = Always
transfer_output_files = ""

+JobFlavour           = "{flavour:s}"
"""
    if "cafalca" in resources:
        job_submit_template += """\
+CAFJob              = True
+AccountingGroup     = "group_u_CMS.CAF.ALCA"
# automatically remove the job if the submitter has no permissions to run a CAF Job
periodic_remove       = !regexp("group_u_CMS.CAF.ALCA", AccountingGroup) && CAFJob =?= True
"""

    if proxy_path is not None:
        job_submit_template += """\
+x509userproxy        = "{proxy:s}"
"""
    job_submit_template += "\nqueue\n"

    job_submit_file = os.path.join(Path, "job.submit")
    with open(job_submit_file, "w") as f:
        f.write(job_submit_template.format(script = os.path.abspath(script),
                                           jobm = os.path.abspath(path),
                                           flavour = job_flavour,
                                           proxy = proxy_path))

    return job_submit_file



parser = argparse.ArgumentParser(
        description="Submit jobs that are setup in local mps database to batch system.",
)
parser.add_argument("maxJobs", type=int, nargs='?', default=1,
                    help="number of Mille jobs to be submitted (default: %(default)d)")
parser.add_argument("-j", "--job-id", dest = "job_id", nargs = "*",
                    help = ("job IDs to be submitted; "
                            "use either 'job<ID>' or directly '<ID>'"))
parser.add_argument("-r", "--resubmit", dest = "resub", default=False, 
                    action="store_true",
                    help = ("resubmit jobs - only works if job IDs are specified"))
parser.add_argument("-a", "--all", dest="allMille", default=False,
                    action="store_true",
                    help = ("submit all setup Mille jobs; "
                            "maxJobs and --job-id are ignored"))
parser.add_argument("-m", "--merge", dest="fireMerge", default=False,
                    action="store_true",
                    help = ("submit all setup Pede jobs; "
                            "maxJobs is ignored, but --job-id is respected"))
parser.add_argument("-f", "--force-merge", dest="forceMerge", default=False,
                    action="store_true",
                    help=("force the submission of the Pede job in case some "+
                          "Mille jobs are not in the OK state"))
parser.add_argument("--force-merge-manual", dest="forceMergeManual", default=False,
                    action="store_true",
                    help=("force the submission of the Pede job in case some "+
                          "Mille jobs are not in the OK state. Unlike --forceMerge "+
                          "this option assumes the user has edited theScript.sh and "+
                          "alignment_merge.py to consistently pick up only the mille "+
                          "output files that exist"))
parser.add_argument("-p", "--forward-proxy", dest="forwardProxy", default=False,
                    action="store_true",
                    help="forward VOMS proxy to batch system")
args = parser.parse_args(sys.argv[1:])


lib = mpslib.jobdatabase()
lib.read_db()

if args.allMille:
    # submit all Mille jobs and ignore 'maxJobs' supplied by user
    args.maxJobs = lib.nJobs
    args.job_id = None

if args.job_id is None and args.resub:
    print("Can only resubmit jobs if job IDs are specified")
    sys.exit(1)


if args.job_id is None:
    job_mask = lib.JOBDIR
else:
    job_mask = []
    for job_id in args.job_id:
        invalid_id = False
        if job_id.startswith("job"): job_mask.append(job_id)
        elif job_id.startswith("m"): job_mask.append("job"+job_id)
        else:
            try:
                job_mask.append(lib.JOBDIR[int(job_id)-1])
            except ValueError:
                invalid_id = True
            except IndexError:
                print("ID provided to '-j/--job-id' is out of range:", job_id)
                sys.exit(1)

        if invalid_id or job_mask[-1] not in lib.JOBDIR:
            print("ID provided to '-j/--job-id' is invalid:", job_id)
            print("'-j/--job-id' requires the IDs to exist and to be of either", end=' ')
            print("of the following formats:")
            print(" - job042")
            print(" - 042")
            print(" - jobm1")
            print(" - m1")
            sys.exit(1)

# build the absolute job directory path (needed by mps_script)
theJobData = os.path.join(os.getcwd(), "jobData")

# set the job name ???????????????????
theJobName = 'mpalign'
if lib.addFiles != '':
    theJobName = lib.addFiles

fire_htcondor = False

# fire the 'normal' parallel Jobs (Mille Jobs)
if not args.fireMerge:
    #set the resources string coming from mps.db
    resources = lib.get_class('mille')

    # "cmscafspec" found in $resources: special cmscaf resources
    if 'cmscafspec' in resources:
        print('\nWARNING:\n  Running mille jobs on cmscafspec, intended for pede only!\n\n')
        resources = '-q cmscafalcamille'
    # "cmscaf" found in $resources
    elif 'cmscaf' in resources:
        # g_cmscaf for ordinary caf queue, keeping 'cmscafspec' free for pede jobs:
        resources = '-q'+resources+' -m g_cmscaf'
    elif "htcondor" in resources:
        fire_htcondor = True
        schedinfo = subprocess.check_output(["myschedd","show"]).decode()
        if 'cafalca' in resources:
            if not 'tzero' in schedinfo:
                print("\nMPS fire: request to use CAF pool which has not been set up. Call `module load lxbatch/tzero` and try again")
                exit(1)
        else:
            if not 'share' in schedinfo:
                print("\nMPS fire: request to use standard pool when CAF pool is set up. Call `module load lxbatch/share` and try again")
                exit(1)
    else:
        resources = '-q '+resources

    nSub = 0 # number of submitted Jobs
    for i in range(lib.nJobs):
        if lib.JOBDIR[i] not in job_mask: continue
        if lib.JOBSTATUS[i] == 'SETUP':
            if nSub < args.maxJobs:
                if args.forwardProxy:
                    forward_proxy(os.path.join(theJobData,lib.JOBDIR[i]))

                # submit a new job with 'bsub -J ...' and check output
                # for some reasons LSF wants script with full path
                if fire_htcondor:
                    Path = os.path.join(theJobData,lib.JOBDIR[i])
                    scriptPath = os.path.join(Path, "theScript.sh")
                    if args.forwardProxy:
                        job_submit_file = write_HTCondor_submit_file_mille(Path, scriptPath, lib,os.path.join(Path,".user_proxy"))
                    else:
                        job_submit_file = write_HTCondor_submit_file_mille(Path, scriptPath, lib)
                    submission = "condor_submit -batch-name %s %s"%\
                          (theJobName, job_submit_file)
                else:
                    submission = 'bsub -J %s %s %s/%s/theScript.sh' % \
                          (theJobName, resources, theJobData, lib.JOBDIR[i])
                print(submission)
                try:
                    result = subprocess.check_output(submission,
                                                     stderr=subprocess.STDOUT,
                                                     shell=True).decode()
                except subprocess.CalledProcessError as e:
                    result = "" # -> check for successful job submission will fail
                #print('      '+result, end=' ')
                print(result)
                result = result.strip()

                # check if job was submitted and updating jobdatabase
                if fire_htcondor:
                    match = re.search(r"1 job\(s\) submitted to cluster (\d+)\.", result)
                else:
                    match = re.search('Job <(\d+)> is submitted', result)
                if match:
                    # need standard format for job number
                    lib.JOBSTATUS[i] = 'SUBTD'
                    lib.JOBID[i] = match.group(1)
                    if fire_htcondor: lib.JOBID[i] += ".0"
                else:
                    print('Submission of %03d seems to have failed: %s' % (lib.JOBNUMBER[i],result), end=' ')
                nSub +=1

        elif args.resub:
            if nSub < args.maxJobs:
                if args.forwardProxy:
                    forward_proxy(os.path.join(theJobData,lib.JOBDIR[i]))

                try:
                    os.remove("%s/%s/HTCJOB" % (theJobData, lib.JOBDIR[i])) 
                except OSError as e:
                    print("Cannot delete file %s/%s/HTCJOB :" % (theJobData,lib.JOBDIR[i]), e.strerror)
                try:
                    os.remove("%s/%s/STDOUT" % (theJobData, lib.JOBDIR[i])) 
                except OSError as e:
                    print("Cannot delete file %s/%s/STDOUT :" % (theJobData,lib.JOBDIR[i]), e.strerror)
                try:
                    os.remove("%s/%s/STDOUT.gz" % (theJobData, lib.JOBDIR[i])) 
                except OSError as e:
                    print("Cannot delete file %s/%s/STDOUT.gz :" % (theJobData,lib.JOBDIR[i]), e.strerror)
                try:
                    os.remove("%s/%s/alignment.log.gz" % (theJobData, lib.JOBDIR[i])) 
                except OSError as e:
                    print("Cannot delete file %s/%s/alignment.log.gz :" % (theJobData,lib.JOBDIR[i]), e.strerror)
                try:
                    os.remove("%s/%s/millePedeMonitor%03d.root" % (theJobData, lib.JOBDIR[i], lib.JOBNUMBER[i]))
                except OSError as e:
                    print("Cannot delete file %s/%s/millePedeMonitor%03d.root :" % (theJobData,lib.JOBDIR[i],lib.JOBNUMBER[i]), e.strerror)

                # submit a new job with 'bsub -J ...' and check output
                # for some reasons LSF wants script with full path
                if fire_htcondor:
                    Path = os.path.join(theJobData,lib.JOBDIR[i])
                    scriptPath = os.path.join(Path, "theScript.sh")
                    if args.forwardProxy:
                        job_submit_file = write_HTCondor_submit_file_mille(Path, scriptPath, lib,os.path.join(Path,".user_proxy"))
                    else:
                        job_submit_file = write_HTCondor_submit_file_mille(Path, scriptPath, lib)
                    submission = "condor_submit -batch-name %s %s"%\
                          (theJobName, job_submit_file)
                else:
                    submission = 'bsub -J %s %s %s/%s/theScript.sh' % \
                          (theJobName, resources, theJobData, lib.JOBDIR[i])
                print(submission)
                try:
                    result = subprocess.check_output(submission,
                                                     stderr=subprocess.STDOUT,
                                                     shell=True).decode()
                except subprocess.CalledProcessError as e:
                    result = "" # -> check for successful job submission will fail
                print('      '+result, end=' ')
                result = result.strip()

                # check if job was submitted and updating jobdatabase
                if fire_htcondor:
                    match = re.search(r"1 job\(s\) submitted to cluster (\d+)\.", result)
                else:
                    match = re.search('Job <(\d+)> is submitted', result)
                if match:
                    # need standard format for job number
                    lib.JOBSTATUS[i] = 'SUBTD'
                    lib.JOBID[i] = match.group(1)
                    if fire_htcondor: lib.JOBID[i] += ".0"
                else:
                    print('Submission of %03d seems to have failed: %s' % (lib.JOBNUMBER[i],result), end=' ')
                nSub +=1


# fire the merge job
else:
    print('fire merge')
    # set the resources string coming from mps.db
    resources = lib.get_class('pede')
    if 'cmscafspec' in resources:
        resources = '-q cmscafalcamille'
    elif "htcondor" in resources:
        fire_htcondor = True
        schedinfo = subprocess.check_output(["myschedd","show"]).decode()
        if 'bigmem' in resources:
            if not 'share' in schedinfo:
                print("\nMPS fire: CAF pool is set up, but request to use high-memory machines which live in the standard pool. Call `module load lxbatch/share` and try again")
                exit(1)
    else:
        resources = '-q '+resources

    if not fire_htcondor:
        # Allocate memory for pede job FIXME check documentation for bsub!!!!!
        resources = resources+' -R \"rusage[mem="%s"]\"' % str(lib.pedeMem) # FIXME the dots? -> see .pl

    # check whether all other jobs are OK
    mergeOK = True
    for i in range(lib.nJobs):
        if lib.JOBSTATUS[i] != 'OK':
            if 'DISABLED' not in lib.JOBSTATUS[i]:
                mergeOK = False
                break

    # loop over merge jobs
    i = lib.nJobs
    while i<len(lib.JOBDIR):
        jobNumFrom1 = i+1
        if lib.JOBDIR[i] not in job_mask:
            i += 1
            continue

        # check if current job in SETUP mode or if forced
        if lib.JOBSTATUS[i] != 'SETUP' and not args.resub:
            print('Merge job %d status %s not submitted.' % \
                  (jobNumFrom1, lib.JOBSTATUS[i]))
        elif not (mergeOK or args.forceMerge or args.forceMergeManual):
            print('Merge job',jobNumFrom1,'not submitted since Mille jobs error/unfinished (Use -m -f to force).')
        elif not args.resub:
            # some paths for clarity
            Path = os.path.join(theJobData,lib.JOBDIR[i])
            backupScriptPath  = os.path.join(Path, "theScript.sh.bak")
            scriptPath        = os.path.join(Path, "theScript.sh")

            # force option invoked:
            if args.forceMerge:

                # make a backup copy of the script first, if it doesn't already exist.
                if not os.path.isfile(backupScriptPath):
                    os.system('cp -p '+scriptPath+' '+backupScriptPath)

                # get the name of merge cfg file -> either the.py or alignment_merge.py
                command  = 'cat '+backupScriptPath+' | grep CONFIG_FILE | head -1 | awk -F"/" \'{print $NF}\''
                mergeCfg = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode()
                mergeCfg = mergeCfg.strip()

                if fire_htcondor:
                    job_submit_file = write_HTCondor_submit_file_pede(Path, scriptPath, mergeCfg, lib)

                # make a backup copy of the cfg
                backupCfgPath  = os.path.join(Path, mergeCfg+".bak")
                cfgPath        = os.path.join(Path, mergeCfg)
                if not os.path.isfile(backupCfgPath):
                    os.system('cp -p '+cfgPath+' '+backupCfgPath)

                # retrieve weights configuration
                with open(os.path.join(Path, ".weights.pkl"), "rb") as f:
                    weight_conf = cPickle.load(f)

                # blank weights
                mps_tools.run_checked(["mps_weight.pl", "-c"])

                # apply weights
                for name,weight in weight_conf:
                    print(" ".join(["mps_weight.pl", "-N", name, weight]))
                    mps_tools.run_checked(["mps_weight.pl", "-N", name, weight])

                # rewrite the mergeCfg using only 'OK' jobs (uses first mille-job as baseconfig)
                inCfgPath = theJobData+'/'+lib.JOBDIR[0]+'/the.py'
                command ='mps_merge.py -w -c '+inCfgPath+' '+Path+'/'+mergeCfg+' '+Path+' '+str(lib.nJobs)
                os.system(command)

                # rewrite theScript.sh using inly 'OK' jobs
                command = 'mps_scriptm.pl -c '+lib.mergeScript+' '+scriptPath+' '+Path+' '+mergeCfg+' '+str(lib.nJobs)+' '+lib.mssDir+' '+lib.mssDirPool
                os.system(command)

            else:
                # restore the backup copy of the script
                if os.path.isfile(backupScriptPath):
                    os.system('cp -pf '+backupScriptPath+' '+scriptPath)

                # get the name of merge cfg file
                command  = "cat "+scriptPath+" | grep '^\s*CONFIG_FILE' | awk -F'=' '{print $2}'"
                mergeCfg = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode()
                command  = 'basename '+mergeCfg
                mergeCfg = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode()
                mergeCfg = mergeCfg.replace('\n','')

                if fire_htcondor:
                    job_submit_file = write_HTCondor_submit_file_pede(Path, scriptPath, mergeCfg, lib)

                # restore the backup copy of the cfg
                backupCfgPath  = Path+'/%s.bak' % mergeCfg
                cfgPath        = Path+'/%s'     % mergeCfg
                if os.path.isfile(backupCfgPath):
                    os.system('cp -pf '+backupCfgPath+' '+cfgPath)

            # end of if/else forceMerge

            # submit merge job
            nMerge = i-lib.nJobs  # 'index' of this merge job
            curJobName = 'm'+str(nMerge)+'_'+theJobName
            if args.forwardProxy: forward_proxy(Path)
            if fire_htcondor:
                submission = ["condor_submit",
                              "-batch-name", curJobName,
                              job_submit_file]
            else:
                submission = ["bsub", "-J", curJobName, resources, scriptPath]
            for _ in range(5):
                try:
                    result = subprocess.check_output(submission, stderr=subprocess.STDOUT).decode()
                    break
                except subprocess.CalledProcessError as e:
                    result = e.output

            print('     '+result, end=' ')
            result = result.strip()

            # check if merge job was submitted and updating jobdatabase
            if fire_htcondor:
                match = re.search(r"1 job\(s\) submitted to cluster (\d+)\.", result)
            else:
                match = re.search('Job <(\d+)> is submitted', result)
            if match:
                lib.JOBSTATUS[i] = 'SUBTD'
                lib.JOBID[i] = match.group(1)
                # need standard format for job number
                if fire_htcondor: lib.JOBID[i] += ".0"
                print("jobid is", lib.JOBID[i])
            else:
                print('Submission of merge job seems to have failed:',result, end=' ')

        elif args.resub:
            # some paths for clarity
            Path = os.path.join(theJobData,lib.JOBDIR[i])
            dircontents = os.listdir(Path)
            for outfile in dircontents:
                if outfile.endswith(".root"):
                    os.remove("%s/%s" %(Path, outfile))
            try:
                os.remove("%s/HTCJOB" % (Path)) 
            except OSError as e:
                print("Cannot delete file %s/HTCJOB :" % (Path), e.strerror)
            try:
                os.remove("%s/STDOUT" % (Path)) 
            except OSError as e:
                print("Cannot delete file %s/STDOUT :" % (Path), e.strerror)
            try:
                os.remove("%s/STDOUT.gz" % (Path)) 
            except OSError as e:
                print("Cannot delete file %s/STDOUT.gz :" % (Path), e.strerror)
            try:
                os.remove("%s/alignment.log.gz" % (Path)) 
            except OSError as e:
                print("Cannot delete file %s/alignment.log.gz :" % (Path), e.strerror)


            backupScriptPath  = os.path.join(Path, "theScript.sh.bak")
            scriptPath        = os.path.join(Path, "theScript.sh")

            # restore the backup copy of the script
            if os.path.isfile(backupScriptPath):
                os.system('cp -pf '+backupScriptPath+' '+scriptPath)

            # get the name of merge cfg file
            command  = "cat "+scriptPath+" | grep '^\s*CONFIG_FILE' | awk -F'=' '{print $2}'"
            mergeCfg = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode()
            command  = 'basename '+mergeCfg
            mergeCfg = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode()
            mergeCfg = mergeCfg.replace('\n','')

            if fire_htcondor:
                job_submit_file = write_HTCondor_submit_file_pede(Path, scriptPath, mergeCfg, lib)

            # restore the backup copy of the cfg
            backupCfgPath  = Path+'/%s.bak' % mergeCfg
            cfgPath        = Path+'/%s'     % mergeCfg
            if os.path.isfile(backupCfgPath):
               os.system('cp -pf '+backupCfgPath+' '+cfgPath)


            # submit merge job
            nMerge = i-lib.nJobs  # 'index' of this merge job
            curJobName = 'm'+str(nMerge)+'_'+theJobName
            if args.forwardProxy: forward_proxy(Path)
            if fire_htcondor:
                submission = ["condor_submit",
                              "-batch-name", curJobName,
                              job_submit_file]
            else:
                submission = ["bsub", "-J", curJobName, resources, scriptPath]
            for _ in range(5):
                try:
                    result = subprocess.check_output(submission, stderr=subprocess.STDOUT).decode()
                    break
                except subprocess.CalledProcessError as e:
                    result = e.output

            print('     '+result, end=' ')
            result = result.strip()

            # check if merge job was submitted and updating jobdatabase
            if fire_htcondor:
                match = re.search(r"1 job\(s\) submitted to cluster (\d+)\.", result)
            else:
                match = re.search('Job <(\d+)> is submitted', result)
            if match:
                lib.JOBSTATUS[i] = 'SUBTD'
                lib.JOBID[i] = match.group(1)
                # need standard format for job number
                if fire_htcondor: lib.JOBID[i] += ".0"
                print("jobid is", lib.JOBID[i])
            else:
                print('Submission of merge job seems to have failed:',result, end=' ')

        i +=1

        # end of while on merge jobs


lib.write_db()
