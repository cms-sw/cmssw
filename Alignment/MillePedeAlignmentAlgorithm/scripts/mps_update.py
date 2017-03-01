#!/usr/bin/env python
import subprocess
import re
import os
import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib

lib = mpslib.jobdatabase()
lib.read_db()

#IDEAS
#change Flag-list to binary True/False -> rename CARE?
#rework string-printing with references

#create a FLAG-list of which entries are to worry about
submittedjobs = 0
FLAG = []					#FLAG[i] = 1  -> don't care
							#FLAG[i] = -1 -> care
#asking 'in' to provide for composits, e.g. DISABLEDFETCH
for i in xrange(len(lib.JOBID)):
	if 'SETUP' in lib.JOBSTATUS[i] or \
	   'DONE'  in lib.JOBSTATUS[i] or \
	   'FETCH' in lib.JOBSTATUS[i] or \
	   'OK'    in lib.JOBSTATUS[i] or \
	   'ABEND' in lib.JOBSTATUS[i] or \
	   'FAIL'  in lib.JOBSTATUS[i]:
		FLAG.append(1)	
	else:
		FLAG.append(-1)
		submittedjobs += 1
print "submitted jobs: ", submittedjobs



#deal with submitted jobs by looking into output of shell('bjobs -l')
if submittedjobs > 0:
	#execute shell command 'bjobs -l' and store output. Include error Messages.	
#	with open ("bjobs_test.txt", "r") as testfile:
#		bjobs = testfile.read().replace('\n', '')
	bjobs = subprocess.check_output('bjobs -l', stderr=subprocess.STDOUT, shell=True)
	bjobs = bjobs.replace('\n','')	

	if bjobs != 'No unfinished job found':	
		bjobs = bjobs.replace(' ','')
		results = bjobs.split('-----------------------')
		#print('\n\n'.join(results))
		#print results
		for line in results:
			line.strip()		#might be unnecessary
			print line
			#extract jobID		
			match = re.search('Job<(\d+?)>,', line)
			if match:
				jobid = int(match.group(1))		# FIXME match.group(0)???????????????????
			#extract job status			
			match = re.search('Status<([A-Z]+?)>', line)
			if match:
				status = match.group(1)
			#extract CPU time
			match = re.search('TheCPUtimeusedis(\d+?)seconds', line)
			cputime = 0
			if match:
				cputime = int(match.group(1))
			print  'out ', jobid, ' ', status, ' ', cputime		#this might fail
						
			#check for disabled Jobs
			theIndex = -1
			disabled = ''
			for k in xrange(len(lib.JOBID)):
				if jobid == lib.JOBID[k]:
					theIndex = k
			if 'DISABLED' in lib.JOBSTATUS[theIndex]:
				disabled = 'DISABLED'

			#continue with next batch job if not found or not interesting
			if theIndex == -1:
				print 'mps_update.py - the job ', jobid,' was not found in the JOBID array'
				continue
			if FLAG[theIndex] == 1:
				continue

			#if deemed interesting (FLAG = -1) update Joblists for mps.db
			lib.JOBSTATUS[theIndex] = disabled+status
			if status == 'RUN' or status == 'DONE':
				if cputime > 0:
					diff = cputime - lib.JOBRUNTIME[theIndex]
					lib.JOBRUNTIME[theIndex] = cputime
					lib.JOBHOST[theIndex] = '+'+str(diff)
					lib.JOBINCR[theIndex] = diff
				else:
					lib.JOBRUNTIME[theIndex] = 0
					lib.JOBINCR[theIndex] = 0
			FLAG[theIndex] = 1;
			print 'set flag of job', theIndex, 'with id', lib.JOBID[theIndex], 'to 1'



#loop over remaining jobs to see whether they are done
for i in xrange(len(lib.JOBID)):

	#check if current job is disabled. Print stuff. Continue if flagged unimportant.
	disabled = ''
	if 'DISABLED' in lib.JOBSTATUS[i]:
		disabled = 'DISABLED'
	print ' DB job ', lib.JOBID[i], 'flag ', FLAG[i]
	if FLAG[i] == 1:
		continue

	#check if job may be done by looking if a folder exists in the project directory.
	#if True  -> jobstatus is set to DONE
	theBatchDirectory = 'LSFJOB_'+str(lib.JOBID[i])
	if os.path.isdir(theBatchDirectory):
		print 'Directory ', theBatchDirectory, 'exists'
		lib.JOBSTATUS[i] = disabled + 'DONE'
	else:
		if 'RUN' in lib.JOBSTATUS[i]:
			print 'WARNING: Job ',i,' in state RUN, neither found by bjobs nor find LSFJOB directory!'

#from Perl-script (dunno): FIXME: check if job not anymore in batch system
#from Perl-script (dunno): might set to FAIL -but probably theBatchDirectory is just somewhere else...



#check for orphaned jobs
for i in xrange(len(lib.JOBID)):
	if FLAG[i] != 1:
		if 'SETUP' in lib.JOBSTATUS[i] or \
		   'DONE'  in lib.JOBSTATUS[i] or \
		   'FETCH' in lib.JOBSTATUS[i] or \
		   'TIMEL' in lib.JOBSTATUS[i] or \
		   'SUBTD' in lib.JOBSTATUS[i]:
			print 'Funny entry index ',i,' job ',lib.JOBID[i],' status ',lib.JOBSTATUS[i]


lib.write_db()
