#!/usr/bin/env python3
#
#
#  Fetch jobs that have DONE status
#  This step is mainly foreseen in case job result files need
#  to be copied from a spool area.
#  On LSF batch, the job output is already in our directories,
#  hence this function does hardly anything except for calling
#  mps_check.py.

from builtins import range
import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib
import os

# update database
os.system("mps_update.py > /dev/null")

lib = mpslib.jobdatabase()
lib.read_db()

# loop over DONE jobs
for i in range(len(lib.JOBID)):
    # check also "FETCH" to recover from possibly failed runs of 'mps_fetch.py'
    if lib.JOBSTATUS[i] in ("DONE", "EXIT", "FETCH", "DISABLEDFETCH"):
        # move the LSF output to /jobData/
        theJobDir = 'jobData/'+lib.JOBDIR[i]
        theBatchDirectory = r"LSFJOB_"+ lib.JOBID[i]

        command = 'mv  %s/* %s/ > /dev/null 2>&1' % (theBatchDirectory, theJobDir)
        os.system(command)
        command = 'rm -rf '+theBatchDirectory
        os.system(command)

        # update the status
        if 'DISABLED' in lib.JOBSTATUS[i]:
            lib.JOBSTATUS[i] = 'DISABLEDFETCH'
        else:
            lib.JOBSTATUS[i] = 'FETCH'

lib.write_db()

# call mps_check
os.system('mps_check.py')

