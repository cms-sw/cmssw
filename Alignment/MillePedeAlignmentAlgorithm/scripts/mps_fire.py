#!/usr/bin/env python
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

import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib
import os
import sys
import subprocess
import re
import argparse

parser = argparse.ArgumentParser(
        description="Submit jobs that are setup in local mps database to batch system.",
)
parser.add_argument("maxJobs", type=int, nargs='?', default=1,
                    help="number of Mille jobs to be submitted (default: %(default)d)")
parser.add_argument("-a", "--all", dest="allMille", default=False,
                    action="store_true",
                    help="submit all setup Mille jobs; maxJobs is ignored")
parser.add_argument("-m", "--merge", dest="fireMerge", default=False,
                    action="store_true",
                    help="submit all setup Pede jobs; maxJobs is ignored")
parser.add_argument("-f", "--force-merge", dest="forceMerge", default=False,
                    action="store_true",
                    help=("force the submission of the Pede job in case some "+
                          "Mille jobs are not in the OK state"))
args = parser.parse_args(sys.argv[1:])


lib = mpslib.jobdatabase()
lib.read_db()

if args.allMille:
    # submit all Mille jobs and ignore 'maxJobs' supplied by user
    args.maxJobs = lib.nJobs

# build the absolute job directory path (needed by mps_script)
theJobData = os.path.join(os.getcwd(), "jobData")

# set the job name ???????????????????
theJobName = 'mpalign'
if lib.addFiles != '':
    theJobName = lib.addFiles

# fire the 'normal' parallel Jobs (Mille Jobs)
if not args.fireMerge:
    #set the resources string coming from mps.db
    resources = lib.get_class('mille')

    # "cmscafspec" found in $resources: special cmscaf resources
    if 'cmscafspec' in resources:
        print '\nWARNING:\n  Running mille jobs on cmscafspec, intended for pede only!\n\n'
        queue = resources
        queue = queue.replace('cmscafspec','cmscaf')
        resources = '-q'+queue+'-R cmscafspec' # FIXME why?
        resources = '-q cmscafalcamille'
    # "cmscaf" found in $resources
    elif 'cmscaf' in resources:
        # g_cmscaf for ordinary caf queue, keeping 'cmscafspec' free for pede jobs:
        resources = '-q'+resources+' -m g_cmscaf'
    else:
        resources = '-q '+resources

    nSub = 0 # number of submitted Jobs
    for i in xrange(lib.nJobs):
        if lib.JOBSTATUS[i] == 'SETUP':
            if nSub < args.maxJobs:
                # submit a new job with 'bsub -J ...' and check output
                # for some reasons LSF wants script with full path
                submission = 'bsub -J %s %s %s/%s/theScript.sh' % \
                      (theJobName, resources, theJobData, lib.JOBDIR[i])
                print submission
                result = subprocess.check_output(submission, stderr=subprocess.STDOUT, shell=True)
                print '      '+result,
                result = result.strip()

                # check if job was submitted and updating jobdatabase
                match = re.search('Job <(\d+)> is submitted', result)
                if match:
                    # need standard format for job number
                    lib.JOBSTATUS[i] = 'SUBTD'
                    lib.JOBID[i] = int(match.group(1))
                else:
                    print 'Submission of %03d seems to have failed: %s' % (lib.JOBNUMBER[i],result),
                nSub +=1

# fire the merge job
else:
    print 'fire merge'
    # set the resources string coming from mps.db
    resources = lib.get_class('pede')
    if 'cmscafspec' in resources:
        queue = resources
        queue = queue.replace('cmscafspec','cmscaf')
        resources = '-q '+queue+' -R cmscafspec' # FIXME why?
        resources = '-q cmscafalcamille'
    else:
        resources = '-q '+resources

    # Allocate memory for pede job FIXME check documentation for bsub!!!!!
    resources = resources+' -R \"rusage[mem="%s"]\"' % str(lib.pedeMem) # FIXME the dots? -> see .pl

    # check whether all other jobs are OK
    mergeOK = True
    for i in xrange(lib.nJobs):
        if lib.JOBSTATUS[i] != 'OK':
            if 'DISABLED' not in lib.JOBSTATUS[i]:
                mergeOK = False
                break

    # loop over merge jobs
    i = lib.nJobs
    while i<len(lib.JOBDIR):
        jobNumFrom1 = i+1

        # check if current job in SETUP mode or if forced
        if lib.JOBSTATUS[i] != 'SETUP':
            print 'Merge job %d status %s not submitted.' % \
                  (jobNumFrom1, lib.JOBSTATUS[i])
        elif not (mergeOK or args.forceMerge):
            print 'Merge job',jobNumFrom1,'not submitted since Mille jobs error/unfinished (Use -m -f to force).'
        else:
            # some paths for clarity
            Path = '%s/%s' % (theJobData,lib.JOBDIR[i])
            backupScriptPath  = Path+'/theScript.sh.bak'
            scriptPath        = Path+'/theScript.sh'

            # force option invoked:
            if args.forceMerge:

                # make a backup copy of the script first, if it doesn't already exist.
                if not os.path.isfile(backupScriptPath):
                    os.system('cp -p '+scriptPath+' '+backupScriptPath)

                # get the name of merge cfg file -> either the.py or alignment_merge.py
                command  = 'cat '+backupScriptPath+' | grep cmsRun | grep "\.py" | head -1 | awk \'{gsub("^.*cmsRun ","");print $1}\''
                mergeCfg = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
                command  = 'basename '+mergeCfg
                mergeCfg = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
                mergeCfg = mergeCfg.replace('\n','')

                # make a backup copy of the cfg
                backupCfgPath  = Path+'/%s.bak' % mergeCfg
                cfgPath        = Path+'/%s'     % mergeCfg
                if not os.path.isfile(backupCfgPath):
                    os.system('cp -p '+cfgPath+' '+backupCfgPath)

                # rewrite the mergeCfg using only 'OK' jobs (uses first mille-job as baseconfig)
                inCfgPath = theJobData+'/'+lib.JOBDIR[0]+'/the.py'
                command ='mps_merge.py -c '+inCfgPath+' '+Path+'/'+mergeCfg+' '+Path+' '+str(lib.nJobs)
                os.system(command)

                # rewrite theScript.sh using inly 'OK' jobs
                command = 'mps_scriptm.pl -c '+lib.mergeScript+' '+scriptPath+' '+Path+' '+mergeCfg+' '+str(lib.nJobs)+' '+lib.mssDir+' '+lib.mssDirPool
                os.system(command)

            else:
                # restore the backup copy of the script
                if os.path.isfile(backupScriptPath):
                    os.system('cp -pf '+backupScriptPath+' '+scriptPath)

                # get the name of merge cfg file
                command  = 'cat '+scriptPath+' | grep cmsRun | grep "\.py" | head -1 | awk \'{gsub("^.*cmsRun ","");print $1}\''
                mergeCfg = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
                command  = 'basename '+mergeCfg
                mergeCfg = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
                mergeCfg = mergeCfg.replace('\n','')

                # restore the backup copy of the cfg
                backupCfgPath  = Path+'/%s.bak' % mergeCfg
                cfgPath        = Path+'/%s'     % mergeCfg
                if os.path.isfile(backupCfgPath):
                    os.system('cp -pf '+backupCfgPath+' '+cfgPath)

            # end of if/else forceMerge

            # submit merge job
            nMerge = i-lib.nJobs  # 'index' of this merge job
            curJobName = 'm'+str(nMerge)+'_'+theJobName
            submission = 'bsub -J %s %s %s' % (curJobName,resources,scriptPath)
            result = subprocess.check_output(submission, stderr=subprocess.STDOUT, shell=True)
            print '     '+result,
            result = result.strip()

            # check if merge job was submitted and updating jobdatabase
            match = re.search('Job <(\d+)> is submitted', result)
            if match:
                # need standard format for job number
                lib.JOBSTATUS[i] = 'SUBTD'
                lib.JOBID[i] = int(match.group(1))
                print 'jobid is',lib.JOBID[i]
            else:
                print 'Submission of merge job seems to have failed:',result,

        i +=1
        # end of while on merge jobs


lib.write_db()



