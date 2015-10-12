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
#  mps_fire.pl [-m[f]] [maxjobs]
#  mps_fire.pl -h

import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib
import os
import sys
import subprocess
import re

lib = mpslib.jobdatabase()
maxJobs = 1
fireMerge = 0
helpwanted = 0
forceMerge = 0
#updateDb = 0

# parse the arguments
for i, arg in enumerate(sys.argv):
	if arg[0] == '-':		
		if 'h' in arg:
			helpwanted = 1		
		if 'm' in arg:
			fireMerge = 1			
			if 'f' in arg:
				forceMerge = 1	
#		elif 'u' in arg:
#			updateDb = 1
	else:
		if i == 1:
			maxJobs = arg
maxJobs = int(maxJobs)

# Option -h ->Print help
if helpwanted != 0:
	print "Usage:\n  mps_fire.pl [-m[f]] [maxjobs]"
	print "\nmaxjobs:       Number of Mille jobs to be submitted (default is one)"
	print "\nKnown options:";
	print "\n  -m   Submit all setup Pede jobs, maxJobs is ignored."
	print "\n  -mf  Force the submission of the Pede job in case"
	print "\n          some Mille jobs are not in the OK state.\n"
	print "\n  -h   This help."
	exit()

lib.read_db()

# build the absolute job directory path (needed by mps_script)
thePwd = subprocess.check_output('pwd', stderr=subprocess.STDOUT, shell=True)
thePwd = thePwd.strip()
theJobData = thePwd+'/jobData'

# set the job name ???????????????????
theJobName = 'mpalign'
if lib.addFiles != '':
	theJobName = lib.addFiles

# fire the 'normal' parallel Jobs (Mille Jobs)
if fireMerge == 0:
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
			if nSub < maxJobs:
				# submit a new job with 'bsub -J ...' and check output
				# for some reasons LSF wants script with full path
				submission = 'bsub -J %s %s %s/%s/theScript.sh' % \
				      (theJobName, resources, theJobData, lib.JOBDIR[i])
				print submission
				result = subprocess.check_output(submission, stderr=subprocess.STDOUT, shell=True)
				print '      '+result
				result = result.strip()
				
				# check if job was submitted and updating jobdatabase 
				match = re.search('Job <(\d+)> is submitted', result)
				if match:
					# need standard format for job number
					lib.JOBSTATUS[i] = 'SUBTD'
					lib.JOBID[i] = int(match.group(1))
					##lib.JOBID[i] = '%07d' % int(match.group(1))
					##print 'jobid is',lib.JOBID[i]
				else:
					print 'Submission of %03d seems to have failed: %s' % (lib.JOBNUMBER[i],result)
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
	mergeOK = 1
	for i in xrange(lib.nJobs):
		if lib.JOBSTATUS[i] != 'OK':
			if 'DISABLED' not in lib.JOBSTATUS[i]:
				mergeOK = 0
				break
	
	# loop over merge jobs
	i = lib.nJobs
	while i<len(lib.JOBDIR):
		jobNumFrom1 = i+1
		
		# check if current job in SETUP mode or if forced
		if lib.JOBSTATUS[i] != 'SETUP':
			print 'Merge job %d status %s not submitted.' % \
			      (jobNumFrom1, lib.JOBSTATUS[i])
		elif (mergeOK != 1) and (forceMerge != 1):
			print 'Merge job',jobNumFrom1,'not submitted since Mille jobs error/unfinished (Use -mf to force).'
		else:
			# some paths for clarity
			Path = '%s/%s' % (theJobData,lib.JOBDIR[i])
			backupScriptPath  = Path+'/theScript.sh.bak'
			scriptPath        = Path+'/theScript.sh'
			
			# force option invoked:			
			if forceMerge == 1:
				
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
				
				# rewrite the mergeCfg using only 'OK' jobs (uses last mille-job as baseconfig)
				inCfgPath = theJobData+'/'+lib.JOBDIR[lib.nJobs]+'/the.py'
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
			print '     '+result
			result = result.strip()
			
			# check if merge job was submitted and updating jobdatabase 
			match = re.search('Job <(\d+)> is submitted', result)
			if match:
				# need standard format for job number
				lib.JOBSTATUS[i] = 'SUBTD'
				lib.JOBID[i] = int(match.group(1))
				##lib.JOBID[i] = '%07d' % int(match.group(1))
				print 'jobid is',lib.JOBID[i]
			else:
				print 'Submission of merge job seems to have failed:',result
			
		i +=1
		# end of while on merge jobs


lib.write_db()



