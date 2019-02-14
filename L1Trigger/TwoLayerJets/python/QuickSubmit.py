import os
import sys
#f=open("listTTBar.txt", "r")
#i=0;
#for i in range(2,1000):
#for i in range(50,150):
for i in range(0,120):
	# Lxplus Batch Job Script
	'''
	f=open("bsub%d.sh" %i, 'w')
	f.write("#!/bin/bash\n")
	f.write("#SBATCH -J AMTrackFit_%d\n" %(i))
    	f.write("#SBATCH -p background-4g\n")
    	f.write("#SBATCH --time=01:30:00\n")
    	f.write("#SBATCH --mem-per-cpu=1000 \n")
    	f.write("#SBATCH -o AMTrackFit_%d.out \n" %(i))
	f.write("#SBATCH -e AMTrackFit_%d.err \n" %(i))
	f.write("cd /fdata/hepx/store/user/rish/CMSSW_9_2_0/src/L1Trigger/TrackFindingTracklet/test/\n")
	f.write("eval `scramv1 runtime -sh`\n")
	sline=line.rstrip('\n')
	f.write("cmsRun L1TrackNtupleMaker_cfg.py %s %d \n" %(sline,i))
	#f.write("cmsRun STUBS_base.py %d \n" %i)
	f.close()
	'''
	#os.system("Qsub -l lnxfarm -o OutPut_LogFile%d -e cmsRun Tracklet_cfg.py %d" %(i,i))	
	#os.system("Qsub -l lnxfarm -o OutPut_LogFile%d -e cmsRun L1TrackPrimaryVertex_cfg.py %d" %(i,i))
	os.system("Qsub -l lnxfarm -o OutPut_LogFile%d -e cmsRun ConfFile_cfg.py %d" %(i,i))
	#os.system("cmsRun L1TrackFastJets_cfg.py %d" %(i))
	#i=i+1
	#os.system("bash bsub%d.sh " %i)
	#os.system('bsub -R "pool>3000" -q 1nd -J job%d < bsub%d.sh ' %(i,i))
