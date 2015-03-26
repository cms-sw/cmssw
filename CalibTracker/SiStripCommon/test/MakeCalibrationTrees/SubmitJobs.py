#!/usr/bin/env python
import urllib
import string
import os
import sys
import commands
import time

#DATASET = '/MinimumBias/Commissioning2015-SiStripCalMinBias-PromptReco-v1/ALCARECO'
DATASET='/Cosmics/Commissioning2015-PromptReco-v1/RECO'
# Set the correct environment
CMSSWDIR='/afs/cern.ch/cms/tracker/sistrvalidation/Calibration/CalibrationTree/CMSSW_7_3_2/src/' # CMSSW version to be used for CRUZET2015
#CMSSWDIR='/afs/cern.ch/cms/tracker/sistrvalidation/Calibration/CalibrationTree/CMSSW_5_3_8_patch3/src' # CMSSW version to be used for 8TeV
RUNDIR  ='/afs/cern.ch/cms/tracker/sistrvalidation/Calibration/CalibrationTree/CMSSW_7_3_2/src/CalibTracker/SiStripCommon/test/MakeCalibrationTrees' # directory where the job will actually run
#CMSSWDIR = '/afs/cern.ch/user/q/querten/workspace/public/CalibTreeUpdate/CMSSW_5_3_8_patch3/src'
#RUNDIR = '/afs/cern.ch/user/q/querten/workspace/public/CalibTreeUpdate/CMSSW_5_3_8_patch3/src/newScript'
CASTORDIR = '/store/group/dpg_tracker_strip/comm_tracker/Strip/Calibration/calibrationtree/CRUZET15'
#CASTORDIR = '/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GRIN' # used for GRIN
#CASTORDIR = '/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR12'
os.environ['PATH'] = os.getenv('PATH')+':/afs/cern.ch/cms/sw/common/'
os.environ['CMS_PATH']='/afs/cern.ch/cms'
os.environ['FRONTIER_PROXY'] = 'http://cmst0frontier.cern.ch:3128'
os.environ['SCRAM_ARCH']='slc6_amd64_gcc481'

initEnv=''
initEnv+='cd ' + CMSSWDIR + ';'
initEnv+='source /afs/cern.ch/cms/cmsset_default.sh' + ';'
initEnv+='eval `scramv1 runtime -sh`' + ';'
initEnv+='cd ' + RUNDIR + ';'   
print initEnv;

#If the script is run without argument, check what was the last run, and check on DBS what are the new runs (with at least 1K events) and run on them
#If on the contrary a runnumber is provided as argument, request the list of input files from DBS in order to run on them and produce calibration trees


if(len(sys.argv)<2): 
   #### FETCH DBS TO FIND ALL RUNS TO PROCCESS

   #PRODUCE A LIST OF FILE WITH ALL NEW RUNS TO BE ANALYZED
   LASTRUN=int(commands.getstatusoutput("tail -n 1 LastRun.txt")[1])
   print('Last analyzed Run: %i' % LASTRUN)

   runs = []
   print ("das_client.py  --limit=9999 --query='run dataset="+DATASET+"  | grep run.run_number>"+str(LASTRUN)+"'")
   results = commands.getstatusoutput(initEnv+"das_client.py  --limit=9999 --query='run dataset="+DATASET+"'")[1].splitlines()

#   print ('dbs search --query="find run,sum(block.numevents) where dataset=/StreamExpressCosmics/Commissioning2013-SiStripCalZeroBias-Express-v1/ALCARECO and run>'+str(LASTRUN)+'"')
#   results = commands.getstatusoutput(initEnv+'dbs search --query="find run,sum(block.numevents) where dataset=/StreamExpressCosmics/Commissioning2013-SiStripCalZeroBias-Express-v1/ALCARECO and run>'+str(LASTRUN)+'"')[1].splitlines()
#   print ('dbs search --query="find run,sum(block.numevents) where dataset=/MinimumBias/*Run*-SiStripCalMinBias-v*/ALCARECO and run>'+str(LASTRUN)+'"')
#   results = commands.getstatusoutput(initEnv+'dbs search --query="find run,sum(block.numevents) where dataset=/MinimumBias/*Run*-SiStripCalMinBias-v*/ALCARECO and run>'+str(LASTRUN)+'"')[1].splitlines()
   results.sort()
   for line in results:
      if(line.startswith('Showing')):continue
      if(len(line)<=0):continue
      linesplit = line.split('   ')
#      if(len(linesplit)<2):continue
      run     = int(line.split('   ')[0])
      if(run<=LASTRUN): continue
#      NEvents = int(line.split('   ')[1])
#      if(NEvents<1000):continue
      runs.append(run)
   ####
   print runs

   #APPENDS TO THE LIST OF RUNS FOR WHICH THE PROCESSING FAILLED IN THE PAST
   FAILLEDRUN=commands.getstatusoutput("cat FailledRun.txt")[1]
   os.system('echo ' + '"   "' + ' > FailledRun.txt') #remove the file since these jobs will be resubmitted now
   for line in FAILLEDRUN.splitlines():
      try:
         run = int(line)
         runs.append(run)
         print "Job running on run " + str(run) + " failed in the past... Resubmitting"
      except:
         continue   
   ####

   #SUBMIT JOB FOR EACH RUN IN THE LIST (see the second part of the script)
   runs.sort()
   runs = list(set(runs)) #remove duplicates
   for run in runs:
      print 'Submitting Run ' + str(run)
      os.system('bsub -q 2nd -J calibTree_' + str(run) +  ' -R "type == SLC6_64 && pool > 30000" ' + ' "python '+RUNDIR+'/SubmitJobs.py '+str(run)+'"' )
      if(run>LASTRUN):os.system('echo ' + str(run) + ' > LastRun.txt')
   ####


elif(sys.argv[1].isdigit()):
   #### RUN ON ONE PARTICULAR RUN (GIVEN IN ARGUMENT)
   PWDDIR  =os.getcwd() #Current Dir
   os.chdir(RUNDIR);

   run = int(sys.argv[1])
   globaltag = 'GR_E_V42' # used in 2015 CRUZET
#   globaltag = 'GR_E_V33A' # used for GRIN
#   globaltag = 'GR_P_V40' # used for 2012
   outfile = 'calibTree_'+str(run)+'.root'

   #reinitialize the afs token, to make sure that the job isn't kill after a few hours of running
   os.system('/usr/sue/bin/kinit -R')

   #GET THE LIST OF FILE FROM THE DATABASE
   files = ''
#   print("das_client.py  --limit=9999 --query='file dataset="+DATASET+" run="+str(run)+"'")
   results = commands.getstatusoutput(initEnv+"das_client.py  --limit=9999 --query='file dataset="+DATASET+" run="+str(run)+"'")
#   print('dbs search --query="find file where dataset=/StreamExpressCosmics/Commissioning2013-SiStripCalZeroBias-Express-v1/ALCARECO and run='+str(run)+'"')
#   results = commands.getstatusoutput(initEnv+'dbs search --query="find file where dataset=/StreamExpressCosmics/Commissioning2013-SiStripCalZeroBias-Express-v1/ALCARECO and run='+str(run)+'"')
#   print('dbs search --query="find file where dataset=/MinimumBias/Run*-SiStripCalMinBias-v*/ALCARECO and run='+str(run)+'"')
#   results = commands.getstatusoutput(initEnv+'dbs search --query="find file where dataset=/MinimumBias/Run*-SiStripCalMinBias-v*/ALCARECO and run='+str(run)+'"')
   if(int(results[0])!=0 or results[1].find('Error:')>=0):
      print results
      os.system('echo ' + str(run) + ' >> FailledRun.txt')
      sys.exit(1)
   filesList = results[1].splitlines();
   for f in filesList:
      if(not f.startswith('/')):continue
      files+="'"+f+"',"
   if(files==''):
      print('no files to process for run '+ str(run))
      sys.exit(0)
   ###


   #BUILD CMSSW CONFIG, START CMSRUN, COPY THE OUTPUT AND CLEAN THE PROJECT
#   os.system('sed -e "s@OUTFILE@/tmp/'+outfile+'@g" -e "s@GLOBALTAG@'+globaltag+'@g" -e "s@FILES@'+files+'@g" '+RUNDIR+'/produceCalibrationTree_template_cfg.py > ConfigFile_'+str(run)+'_cfg.py')
   os.system('sed -e "s@OUTFILE@'+PWDDIR+'/'+outfile+'@g" -e "s@GLOBALTAG@'+globaltag+'@g" -e "s@FILES@'+files+'@g" '+RUNDIR+'/produceCalibrationTree_template_cfg.py > ConfigFile_'+str(run)+'_cfg.py')
   exit_code = os.system(initEnv+'cmsRun ConfigFile_'+str(run)+'_cfg.py')
   if(int(exit_code)!=0):
      print("Job Failed with ExitCode "+str(exit_code))
      os.system('echo ' + str(run) + ' >> FailledRun.txt')
   else:
      os.system('cmsRm ' + CASTORDIR+'/'+outfile) #make sure that the file is overwritten
      FileSizeInKBytes =commands.getstatusoutput('ls  -lth --block-size=1024 '+PWDDIR+'/'+outfile)[1].split()[4]
      if(int(FileSizeInKBytes)>50): 
         print("Preparing for stageout of " + PWDDIR+'/'+outfile + ' on ' + CASTORDIR+'/'+outfile + '.  The file size is %d KB' % int(FileSizeInKBytes))
         os.system('cmsStageOut '+PWDDIR+'/'+outfile + ' ' + CASTORDIR+'/'+outfile)
         os.system('cmsLs ' + CASTORDIR+'/'+outfile)
      else:
         print('File size is %d KB, this is under the threshold --> the file will not be transfered on EOS' % int(FileSizeInKBytes))
   os.system('ls -lth '+PWDDIR+'/'+outfile)
   os.system('rm -f '+PWDDIR+'/'+outfile)
   os.system('rm -f ConfigFile_'+str(run)+'_cfg.py')
   os.system('cd /afs/cern.ch/user/c/cmstkmtc')
   os.system('rm -rf LSFJOB_${LSB_JOBID}')
   ###

elif(sys.argv[1]=="--corrupted"):
   #### FIND ALL CORRUPTED FILES ON CASTOR AND MARK THEM AS FAILLED RUN

   calibTreeList = ""
   print("Get the list of calibTree from" + CASTORDIR + ")")
   calibTreeInfo = commands.getstatusoutput("cmsLs " + CASTORDIR)[1].split('\n');
   NTotalEvents = 0;
   run = 0
   for info in calibTreeInfo:
      subParts = info.split();
      run = int(subParts[8].replace("calibTree_","").replace(".root",""))
      file = "root://eoscms//eos/cms"+CASTORDIR+"/"+subParts[5] 
      print("Checking " + file)
      results = commands.getstatusoutput(initEnv+'root -l -b -q ' + file)
      if(len(results[1].splitlines())>2):
         print(results[1]);
         print("add " + str(run) + " to the list of failled runs")
         os.system('echo ' + str(run) + ' >> FailledRun.txt')

else:
   #### UNKNOWN CASE
   print "unknown argument: make sure you know what you are doing?"
