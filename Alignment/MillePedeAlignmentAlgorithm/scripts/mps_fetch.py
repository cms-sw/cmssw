#!/usr/bin/env python
#
#
#  Fetch jobs that have DONE status
#  This step is mainly foreseen in case job result files need 
#  to be copied from a spool area.
#  On LSF batch, the job output is already in our directories,
#  hence this function does hardly anything except for calling 
#  mps_check.py.

import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib
import os

# update database
os.system("mps_update.py > /dev/null")

lib = mpslib.jobdatabase()
lib.read_db()

# loop over DONE jobs
for i in xrange(len(lib.JOBID)):
	if 'DONE' in lib.JOBSTATUS[i]:
		# move the LSF output to /jobData/
		theJobDir = 'jobData/'+lib.JOBDIR[i]
		theBatchDirectory = 'LSFJOB\_%d' % lib.JOBID[i]
		
		command = 'mv  %s/* %s/' % (theBatchDirectory, theJobDir)
		os.system(command)
		command = 'rmdir '+theBatchDirectory
		os.system(command)
		
		# update the status
		if 'DISABLED' in lib.JOBSTATUS[i]:
			lib.JOBSTATUS[i] = 'DISABLEDFETCH'
		else:
			lib.JOBSTATUS[i] = 'FETCH'
			
lib.write_db()

# call mps_check
os.system('mps_check.py')
		
