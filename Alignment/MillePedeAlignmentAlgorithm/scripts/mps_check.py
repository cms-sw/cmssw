#!/usr/bin/env python
#
#  This script checks outputs from jobs that have FETCH status and updates if errors occured
#  -> check STDOUT files
#  -> check cmsRun.out
#  -> check alignment.log
#  -> check if millebinaries are on eos
#  -> check pede.dump
#  -> check millepede.log
#  -> check millepede.end
#
#  It also retirieves number of events from alignment.log and cputime from STDOUT

from __future__ import print_function
import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib
import subprocess
import re
import os

lib = mpslib.jobdatabase()
lib.read_db()

# create a list of eos ls entries containing files on eos binary store
command = ["ls", "-l", os.path.join(lib.mssDir, "binaries")]
try:
    eoslsoutput = subprocess.check_output(command, stderr=subprocess.STDOUT).split('\n')
except subprocess.CalledProcessError:
    eoslsoutput = ""

# loop over FETCH jobs
for i in xrange(len(lib.JOBID)):
    # FIXME use bools?
    batchSuccess = 0
    batchExited = 0
    finished = 0
    endofjob = 0
    eofile = 1  # do not deal with timel yet
    timel = 0
    killed = 0
    ioprob = 0
    fw8001 = 0
    tooManyTracks = 0
    segviol = 0
    rfioerr = 0
    quota = 0
    nEvent = 0
    cputime = -1
    pedeAbend = 0
    pedeLogErr = 0
    pedeLogWrn = 0
    exceptionCaught = 0
    timeout = 0
    cfgerr = 0
    emptyDatErr = 0
    emptyDatOnFarm = 0
    cmdNotFound = 0
    insuffPriv = 0
    quotaspace = 0

    kill_reason = None
    pedeLogErrStr = ""
    pedeLogWrnStr = ""
    remark = ""

    disabled = "";
    if 'DISABLED' in lib.JOBSTATUS[i]:
        disabled = 'DISABLED'

    if 'FETCH' in lib.JOBSTATUS[i]:

        # open the STDOUT file
        stdOut = 'jobData/'+lib.JOBDIR[i]+'/STDOUT'
        # unzip the STDOUT file if necessary
        if os.access(stdOut+'.gz', os.R_OK):
            os.system('gunzip '+stdOut+'.gz')

        try:
            with open(stdOut, "r") as STDFILE:
                # scan records in input file.
                # use regular expression to search. re.compile needed for options re.M and re.I
                # re.M=re.MULTILINE enables matching of newline char
                # re.I=re.IGNORECASE makes matching case-insensitive.
                for line in STDFILE:
                    if re.search(re.compile('Unable to access quota space',re.M|re.I), line):
                        quotaspace = 1
                    if re.search(re.compile('Unable to get quota space',re.M|re.I), line):
                        quotaspace = 1
                    if re.search(re.compile('Disk quota exceeded',re.M|re.I), line):
                        quotaspace = 1
                    if re.search(re.compile('CERN report: Job Killed',re.M), line):
                        killed = 1
                    if re.search(re.compile('Job finished',re.M), line):
                        finished = 1
                    if re.search(re.compile('connection timed out',re.M), line):
                        timeout = 1
                    if re.search(re.compile('ConfigFileReadError',re.M), line):
                        cfgerr = 1
                    if re.search(re.compile('0 bytes transferred',re.M), line):
                        emptyDatOnFarm = 1
                    if re.search(re.compile('command not found',re.M), line):
                        cmdNotFound = 1
                    # AP 26.11.2009 Insufficient privileges to rfcp files
                    if re.search(re.compile('stage_put: Insufficient user privileges',re.M), line):
                        insuffPriv = 1
                    # AP 05.11.2015 Extract cpu-time.
                    # STDOUT doesn't contain NCU anymore. Now KSI2K and HS06 seconds are displayed.
                    # The ncuFactor is calculated from few samples by comparing KSI2K seconds with
                    # CPU time from email.
                    match = re.search(re.compile('This process used .+?(\d+) KSI2K seconds',re.M|re.I), line)
                    if match:
                        cpuFactor = 2.125
                        cputime = int(round(int(match.group(1))/cpuFactor)) # match.group(1) is the matched digit

            # gzip it afterwards:
            print('gzip -f '+stdOut)
            os.system('gzip -f '+stdOut)
        except IOError as e:
            if e.args == (2, "No such file or directory"):
                print("mps_check.py cannot find", stdOut, "to test")
            else:
                raise

        # check HTCondor log file
        try:
            log_file = os.path.join("jobData", lib.JOBDIR[i], "HTCJOB")
            condor_log = subprocess.check_output(["condor_q", lib.JOBID[i],
                                                  "-userlog", log_file,
                                                  "-af",
                                                  "RemoteSysCpu",
                                                  "JobStatus",
                                                  "RemoveReason"],
                                                 stderr = subprocess.STDOUT)
            condor_log = condor_log.split()

            cputime = int(round(float(condor_log[0])))

            if condor_log[1] == "3": # JobStatus == Removed
                killed = 1
                kill_reason = " ".join(condor_log[2:])

        except subprocess.CalledProcessError as e:
            pass


        # GF: This file is not produced (anymore...) -> check for existence and read-access added
        eazeLog = 'jobData/'+lib.JOBDIR[i]+'/cmsRun.out'
        if os.access(eazeLog, os.R_OK):
            # open the input file
            with open(eazeLog, "r") as INFILE:
                # scan records in input file
                for line in INFILE:
                    # check if end of file has been reached
                    if re.search(re.compile('\<StorageStatistics\>',re.M), line):
                        eofile = 1
                    if re.search(re.compile('Time limit reached\.',re.M), line):
                        timel = 1
                    if re.search(re.compile('gives I\/O problem',re.M), line):
                        ioprob = 1
                    if re.search(re.compile('FrameworkError ExitStatus=[\'\"]8001[\'\"]',re.M), line):
                        fw8001 = 1
                    if re.search(re.compile('too many tracks',re.M), line):
                        tooManyTracks = 1
                    if re.search(re.compile('segmentation violation',re.M), line):
                        segviol = 1
                    if re.search(re.compile('failed RFIO error',re.M), line):
                        rfioerr = 1
                    if re.search(re.compile('Request exceeds quota',re.M), line):
                        quota = 1

        # if there is an alignment.log[.gz] file, check it as well
        eazeLog = 'jobData/'+lib.JOBDIR[i]+'/alignment.log'
        logZipped = 'no'
        # unzip the logfile if necessary
        if os.access(eazeLog+'.gz', os.R_OK):
            os.system('gunzip '+eazeLog+'.gz')
            logZipped = 'true'

        if os.access(eazeLog, os.R_OK):   # access to alignment.log
            # open the input file
            with open(eazeLog,'r') as INFILE:
                # scan records in input file
                for line in INFILE:
                    # check if end of file has been reached
                    if re.search(re.compile('\<StorageStatistics\>',re.M), line):
                        eofile = 1
                    if re.search(re.compile('EAZE\. Time limit reached\.',re.M), line):
                        timel = 1
                    if re.search(re.compile('GAF gives I\/O problem',re.M), line):
                        ioprob = 1
                    if re.search(re.compile('FrameworkError ExitStatus=[\'\"]8001[\'\"]',re.M), line):
                        fw8001 = 1
                    if re.search(re.compile('too many tracks',re.M), line):
                        tooManyTracks = 1
                    if re.search(re.compile('segmentation violation',re.M), line):
                        segviol = 1
                    if re.search(re.compile('failed RFIO error',re.M), line):
                        rfioerr = 1
                    if re.search(re.compile('Request exceeds quota',re.M), line):
                        quota = 1
                    # check for newer (e.g. CMSSW_5_1_X) and older CMSSW:
                    if re.search(re.compile('Fatal Exception',re.M), line):
                        exceptionCaught = 1
                    if re.search(re.compile('Exception caught in cmsRun',re.M), line):
                        exceptionCaught = 1
                    # AP 07.09.2009 - Check that the job got to a normal end
                    if re.search(re.compile('AlignmentProducerAsAnalyzer::endJob\(\)',re.M), line):
                        endofjob = 1
                    if re.search(re.compile('FwkReport            -i main_input:sourc',re.M), line):
                        array = line.split()
                        nEvent = int(array[5])
                    if nEvent==0 and re.search(re.compile('FwkReport            -i PostSource',re.M), line):
                        array = line.split()
                        nEvent = int(array[5])
                    # AP 31.07.2009 - To read number of events in CMSSW_3_2_2_patch2
                    if nEvent==0 and re.search(re.compile('FwkReport            -i AfterSource',re.M), line):
                        array = line.split()
                        nEvent = int(array[5])

            if logZipped == 'true':
                os.system('gzip '+eazeLog)

        else:   # no access to alignment.log
            print('mps_check.py cannot find',eazeLog,'to test')
            # AP 07.09.2009 - The following check cannot be done: set to 1 to avoid fake error type
            endofjob = 1

        # for mille jobs checks that milleBinary file is not empty
        if i<lib.nJobs:  # mille job!
            milleOut = 'milleBinary%03d.dat' % (i+1)
            # from Perl, should be deprecated because of cmsls and nsls
            #(not translated to python, yes I'm lazy... use subprocess.checkout if needed):
            #$mOutSize = `nsls -l $mssDir | grep $milleOut | head -1 | awk '{print \$5}'`;
            #$mOutSize = `cmsLs -l $mssDir | grep $milleOut | head -1 | awk '{print \$2}'`;
            mOutSize = 0
            for line in eoslsoutput:
                if milleOut in line:
                    columns = line.split()
                    mOutSize = columns[4] # 5th column = size
            if not (mOutSize>0):
                emptyDatErr = 1

        # merge jobs: additional checks for merging job
        else:
            # if there is a pede.dump file check it as well
            eazeLog = 'jobData/'+lib.JOBDIR[i]+'/pede.dump'
            if os.access(eazeLog+'.gz', os.R_OK):
                # unzip - but clean before and save to tmp
                os.system('rm -f /tmp/pede.dump')
                os.system('gunzip -c '+eazeLog+'.gz > /tmp/pede.dump')
                eazeLog = '/tmp/pede.dump'
            if os.access(eazeLog, os.R_OK):
                with open(eazeLog, "r") as INFILE: # open pede.dump
                    # scan records in INFILE
                    pedeAbend = 1
                    usedPedeMem = 0.
                    for line in INFILE:
                        # check if pede has reached its normal end
                        if re.search(re.compile('Millepede II.* ending',re.M), line):
                            pedeAbend = 0
                        # extract memory usage
                        match = re.search(re.compile('Peak dynamic memory allocation: (.+) GB',re.I), line)
                        if match:
                            mem = match.group(1)
                            mem = re.sub('\s', '', mem)
                            # if mem is a float
                            if re.search(re.compile('^\d+\.\d+$',re.M), mem):
                                usedPedeMem = float(mem)
                            else:
                                print('mps_check.py: Found Pede peak memory allocation but extracted number is not a float:',mem)

                # check memory usage
                # no point in asking if lib.pedeMem is defined. Initialized as lib.pedeMem=-1
                if lib.pedeMem > 0 and usedPedeMem > 0.:
                    memoryratio = usedPedeMem /(lib.pedeMem/1024.)
                    # print a warning if more than approx. 4 GB have been
                    # requested of which less than 75% are used by Pede
                    if lib.pedeMem > 4000 and memoryratio < 0.75 :
                        msg = ("Warning: {0:.2f} GB of memory for Pede "
                               "requested, but only {1:.1f}% of it has been "
                               "used! Consider to request less memory in order "
                               "to save resources.")
                        print(msg.format(lib.pedeMem/1024.0, memoryratio*100))
                    elif memoryratio > 1 :
                        msg = ("Warning: {0:.2f} GB of memory for Pede "
                               "requested, but {1:.1f}% of this has been "
                               "used! Consider to request more memory to avoid "
                               "premature removal of the job by the admin.")
                        print(msg.format(lib.pedeMem/1024.0, memoryratio*100))
                    else:
                        msg = ("Info: Used {0:.1f}% of {1:.2f} GB of memory "
                               "which has been requested for Pede.")
                        print(msg.format(memoryratio*100, lib.pedeMem/1024.0))


                # clean up /tmp/pede.dump if needed
                if eazeLog == '/tmp/pede.dump':
                    os.system('rm /tmp/pede.dump')

            # pede.dump not found or no read-access
            else:
                print('mps_check.py cannot find',eazeLog,'to test')

            # if there is a millepede.log file, check it as well
            eazeLog = 'jobData/'+lib.JOBDIR[i]+'/millepede.log'
            logZipped = 'no'
            if os.access(eazeLog+'.gz', os.R_OK):
                os.system('gunzip '+eazeLog+'.gz')
                logZipped = 'true'

            if os.access(eazeLog, os.R_OK):
                # open log file
                with open(eazeLog, "r") as INFILE:
                    # scan records in input file
                    for line in INFILE:
                        # Checks for Pede Errors
                        if re.search(re.compile('step no descending',re.M), line):
                            pedeLogErr = 1
                            pedeLogErrStr += line
                        if re.search(re.compile('Constraint equation discrepancies:',re.M), line):
                            pedeLogErr = 1
                            pedeLogErrStr += line
                        # AP 07.09.2009 - Checks for Pede Warnings:
                        if re.search(re.compile('insufficient constraint equations',re.M), line):
                            pedeLogWrn = 1
                            pedeLogWrnStr += line

                if logZipped == 'true':
                    os.system('gzip '+eazeLog)
            else:
                print('mps_check.py cannot find',eazeLog,'to test')


            # check millepede.end -- added F. Meier 03.03.2015
            eazeLog = 'jobData/'+lib.JOBDIR[i]+'/millepede.end'
            logZipped = 'no'
            if os.access(eazeLog+'.gz', os.R_OK):
                os.system('gunzip'+eazeLog+'.gz')
                logZipped = 'true'

            if os.access(eazeLog, os.R_OK):
                # open log file
                with open(eazeLog, "r") as INFILE:
                    # scan records in input file
                    for line in INFILE:
                        # Checks for the output code. 0 is OK, 1 is WARN, anything else is FAIL
                        # searches the line for a number with or without a sign
                        match = re.search(re.compile('([-+]?\d+)',re.M), line)
                        if match:
                            if int(match.group(1)) == 1:
                                pedeLogWrn = 1
                                pedeLogWrnStr += line
                            elif int(match.group(1)) != 0:
                                pedeLogErr = 1
                                pedeLogErrStr += line
                if logZipped == 'true':
                    os.system('gzip '+eazeLog)
            else:
                print('mps_check.py cannot find',eazeLog,'to test')

        # end of merge job checks
        # evaluate Errors:
        farmhost = ' '

        okStatus = 'OK'
        if not eofile == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'did not reach end of file')
            okStatus = 'ABEND'
        if quotaspace == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'had quota space problem')
            okStatus = 'FAIL'
            remark = 'eos quota space problem'
        if ioprob == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'had I/O problem')
            okStatus = 'FAIL'
        if fw8001 == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'had Framework error 8001 problem')
            remark = 'fwk error 8001'
            okStatus = 'FAIL'
        if timeout == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'had connection timed out problem')
            remark = 'connection timed out'
        if cfgerr == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'had config file error')
            remark = 'cfg file error'
            okStatus = 'FAIL'
        if killed == 1:
            guess = " (probably time exceeded)" if kill_reason is None else ":"
            print(lib.JOBDIR[i], lib.JOBID[i], "Job killed" + guess)
            if kill_reason is not None: print(kill_reason)
            remark = "killed";
            okStatus = "FAIL"
        if timel == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'ran into time limit')
            okStatus = 'TIMEL'
        if tooManyTracks == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'too many tracks')
        if segviol == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'SEGVIOL encountered')
            remark = 'seg viol'
            okStatus = 'FAIL'
        if rfioerr == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'RFIO error encountered')
            remark = 'rfio error'
            okStatus = 'FAIL'
        if quota == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'Request exceeds quota')
        if exceptionCaught == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'Exception caught in cmsrun')
            remark = 'Exception caught'
            okStatus = 'FAIL'
        if emptyDatErr == 1:
            print('milleBinary???.dat file not found or empty')
            remark = 'empty milleBinary'
            if emptyDatOnFarm > 0:
                print('...but already empty on farm so OK (or check job',i+1,'yourself...)')
            else:
                okStatus = 'FAIL'
        if cmdNotFound == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'Command not found')
            remark = 'cmd not found'
            okStatus = 'FAIL'
        if insuffPriv == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'Insufficient privileges to rfcp files')
            remark = 'Could not rfcp files'
            okStatus = 'FAIL'
        if pedeAbend == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'Pede did not end normally')
            remark = 'pede failed'
            okStatus = 'FAIL'
        if pedeLogErr == 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'Problems in running Pede:')
            print(pedeLogErrStr)
            remark = 'pede error'
            okStatus = 'FAIL'
        if pedeLogWrn == 1:
            # AP 07.09.2009 - Reports Pede Warnings (but do _not_ set job status to FAIL)
            print(lib.JOBDIR[i],lib.JOBID[i],'Warnings in running Pede:')
            print(pedeLogWrnStr)
            remark = 'pede warnings'
            okStatus = 'WARN'
        if endofjob != 1:
            print(lib.JOBDIR[i],lib.JOBID[i],'Job not ended')
            remark = 'job not ended'
            okStatus = 'FAIL'

        # print warning line to stdout
        if okStatus != "OK":
            print(lib.JOBDIR[i],lib.JOBID[i],' -------- ',okStatus)

        # update number of events
        lib.JOBNEVT[i] = nEvent
        # udate Jobstatus
        lib.JOBSTATUS[i] = disabled+okStatus
        # update cputime
        lib.JOBRUNTIME[i] = cputime
        # update remark
        lib.JOBREMARK[i] = remark
        # update host
        ##lib.JOBHOST[i] = farmhost

lib.write_db()

