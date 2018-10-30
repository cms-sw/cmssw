#!/usr/bin/env python

from __future__ import print_function
import os,sys
import getopt
import commands
import time
import ROOT
import urllib
import string
import optparse
import dataCert

def numberOfEvents(file,mode):
   if mode=='':
      mode="StdBunch0T"
   rootfile = ROOT.TFile.Open(file,'read')
   tree = ROOT.TTree()
   rootfile.GetObject("gainCalibrationTree%s/tree"%mode,tree)
   NEntries = tree.GetEntries()
   rootfile.Close()
   print(file +' --> '+str(NEntries))
   print("gainCalibrationTree%s/tree"%mode)
   return NEntries	


PCLDATASET = "/StreamExpress/Run2017*-PromptCalibProdSiStripGains__AAG__-Express-v*/ALCAPROMPT"
CALIBTREEPATH = '/store/group/dpg_tracker_strip/comm_tracker/Strip/Calibration/calibrationtree/GR17__AAG__' #used if usePCL==False

runValidation = 273555
runsToVeto = [272935, 273290, 273294, 273295, 273296, 273531,
              273537, 273526, 273523, 273514, 273589,  # Low PU
              273588, 273586, 273581, 273580, 273579, 273522, 273521, 273520, 273512, 273511, 273510, 273509,  #VDM
              275326, 275656, 276764, 275765, 275769, 275783, 275825, 275829, 275838
]

#read arguments to the command line
#configure
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-f', '--firstRun'   ,    dest='firstRun'           , help='first run to process (-1 --> automatic)'  , default='-1')
parser.add_option('-l', '--lastRun'    ,    dest='lastRun'            , help='last run to process (-1 --> automatic)'   , default='-1')
parser.add_option('-P', '--publish'    ,    dest='publish'            , help='publish the results'                      , default='True')
parser.add_option('-p', '--pcl'        ,    dest='usePCL'             , help='use PCL output instead of calibTree'      , default='False')
parser.add_option('-m', '--mode'       ,    dest='calMode'            , help='select the statistics type'      , default='AagBunch')
(opt, args) = parser.parse_args()

scriptDir = os.getcwd()
globaltag = "92X_dataRun2_Express_v2"
firstRun = int(opt.firstRun)
lastRun  = int(opt.lastRun)
calMode  = str(opt.calMode) if not str(opt.calMode)=='' else "AagBunch" # Set default to AAG
PCLDATASET = PCLDATASET.replace("__AAG__","") if calMode.lower()=="stdbunch" else PCLDATASET.replace("__AAG__","AAG")
CALIBTREEPATH = CALIBTREEPATH.replace("__AAG__","") if calMode.lower()=="stdbunch" else CALIBTREEPATH.replace("__AAG__","_Aag")
MC=""
publish = (opt.publish=='True')
mail = ""
automatic = True;
usePCL = (opt.usePCL=='True')
minNEvents = 3000      # minimum events for a run to be accepted for the gain payload computation
maxNEvents = 3000000   # maximum events allowed in a gain payload computation

if(firstRun!=-1 or lastRun!=-1): automatic = False


DQM_dir = "AlCaReco/SiStripGains" if "AagBunch" not in opt.calMode else "AlCaReco/SiStripGainsAAG"

print()
print()
print("Gain payload computing configuration")
print("  firstRun = " +str(firstRun))
print("  lastRun  = " +str(lastRun))
print("  publish  = " +str(publish))
print("  usePCL   = " +str(usePCL))
print("  calMode  = " + calMode)
print("  DQM_dir  = " + DQM_dir)
print()


#go to parent directory = test directory
os.chdir("..");

#identify last run of the previous calibration
if(firstRun<=0):
   out = commands.getstatusoutput("ls /afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/ParticleGain/ | grep Run_ | tail -n 1");
   firstRun = int(out[1].split('_')[3])+1
   print("firstRun set to " +str(firstRun))
   print()

initEnv='cd ' + os.getcwd() + ';'
initEnv+='source /afs/cern.ch/cms/cmsset_default.sh' + ';'
initEnv+='eval `scramv1 runtime -sh`' + ';'


#Get List of Files to process:
NTotalEvents = 0;
run = 0
FileList = ""

dataCertInfo = dataCert.get()
print("Loaded certification info. Last update : %s"%dataCertInfo["Last update"]) 

lastGoodRun = -1
if(usePCL==True):
   print("Get the list of PCL output files from DAS")
   print(initEnv+"das_client.py  --limit=9999 --query='dataset=%s'"%PCLDATASET)
   dasOutput = commands.getstatusoutput(initEnv+"das_client.py  --limit=9999 --query='dataset=%s'"%PCLDATASET)[1]
   datasets = [ line for line in dasOutput.splitlines()
                if not line.startswith('Showing') and 'SCRAM fatal' not in line and len(line)>0 ]
   print(datasets)
   if len( datasets)==0 or 'Error' in " ".join(datasets):
       print("Issues in gathering the dataset names, please check the command and the query")
       print("***** DAS OUTPUT *****")
       print(dasOutput)
       print("**********************")
       exit (0) 

   runs = []
   for dataset in datasets:
       runs += [ (dataset, line) for line in commands.getstatusoutput(initEnv+
                                         "das_client.py  --limit=9999 --query='run dataset=%s'"%dataset)[1].splitlines()
                      if not line.startswith('Showing') and 'SCRAM fatal' not in line and len(line)>0 ]
   if len( runs )==0 or 'Error' in " ".join(datasets):
       print("Issues in gathering the run numbers, please check the command and the query")
       exit (0)
   sorted( runs, key=lambda x: x[1])

   for dataset, run_number in runs:
      run  = int(run_number)
      if(run<firstRun or run in runsToVeto):continue
      if(lastRun>0 and run>lastRun):continue      
      if not dataCert.checkRun(run,dataCertInfo):
         print("Skipping...")
         continue
      lastGoodRun = run
      sys.stdout.write( 'Gathering infos for RUN %i:  ' % run )

      #check the events available for this run
      NEventsDasOut = [ line for line in commands.getstatusoutput(initEnv+
       "das_client.py  --limit=9999 --query='summary dataset=%s run=%i | grep summary.nevents'"%(dataset,run))[1].splitlines()
                        if not line.startswith('Showing') and 'SCRAM fatal' not in line and len(line)>0 ][-1]
      if(not NEventsDasOut.isdigit() ):
         print ("cannot retrieve the number of events from das, SKIPPING")
         #print NEventsDasOut
         continue

      if(FileList==""):firstRun=run;
      NEvents = int(NEventsDasOut)

      if(NEvents<=minNEvents):
         print ("only %i events in this run, SKIPPING" % NEvents)
         continue

      FileList+="#run=" + str(run) + " -->  NEvents="+str(NEvents/1000).rjust(8)+"K\n"
      resultsFiles = [ line for line in commands.getstatusoutput(initEnv+
                       "das_client.py  --limit=9999 --query='file dataset=%s run=%i'"%(dataset,run))[1].splitlines()
                       if not line.startswith('Showing') and 'SCRAM fatal' not in line and len(line)>0 ]
      if len(resultsFiles)==0 or 'Error' in " ".join(resultsFiles):
         print ("cannot retrieve the list of files from das, SKIPPING")
         #print resultsFiles
         continue

      for file in resultsFiles: FileList+='calibTreeList.extend(["'+file+'"])\n'
      NTotalEvents += NEvents;

      print("including %s events in gain processing, cumulative total=%i" % (str(NEvents).rjust(7),NTotalEvents))
      if(automatic==True and NTotalEvents >= maxNEvents):break;

else:
   print("Get the list of calibTree from castor (eos ls " + CALIBTREEPATH + ")")
   calibTreeInfo = commands.getstatusoutput("eos ls -l "+CALIBTREEPATH)[1].split('\n');
   print(calibTreeInfo)
   # collect the list of runs and file size
   # calibTreeInfo.split()[8] - file name
   # calibTreeInfo.split()[4] - file size
   info_list = [(i.split()[8],
                 int( i.split()[8].replace("calibTree_","").replace(".root","").split('_')[0] ),
                 int( i.split()[4] )/1048576 ) for i in calibTreeInfo]
   info_list.sort( key=lambda tup: tup[1] )

   print("Check the number of events available")
   for info in info_list:
      if(len(info)<1):continue;
      size = info[2]
      if(size < 10): continue	#skip file<10MB
      run = info[1]
      if(run<firstRun or run in runsToVeto):continue
      if(lastRun>0 and run>lastRun):continue
      if not dataCert.checkRun(run,dataCertInfo):
         print("Skipping...")
         continue
      if run<295310:
         print("Skipping...")
         continue
      lastGoodRun = run
      NEvents = numberOfEvents("root://eoscms//eos/cms"+CALIBTREEPATH+'/'+info[0],calMode);	
      if(NEvents<=3000):continue #only keep runs with at least 3K events
      if(FileList==""):firstRun=run;
      FileList += 'calibTreeList.extend(["root://eoscms//eos/cms'+CALIBTREEPATH+'/'+info[0]+'"]) #' + str(size).rjust(6)+'MB  NEvents='+str(NEvents/1000).rjust(8)+'K\n'
      NTotalEvents += NEvents;
      print("Current number of events to process is " + str(NTotalEvents))
      if(automatic==True and NTotalEvents >= maxNEvents):break;

if lastGoodRun < 0:
   print("No good run to process.")
   sys.exit()
if(lastRun<=0):lastRun = lastGoodRun

print("RunRange=[" + str(firstRun) + "," + str(lastRun) + "] --> NEvents=" + str(NTotalEvents/1000)+"K")

if(automatic==True and NTotalEvents<2e6):	#ask at least 2M events to perform the calibration
	print('Not Enough events to run the calibration')
        os.system('echo "Gain calibration postponed" | mail -s "Gain calibration postponed ('+str(firstRun)+' to '+str(lastRun)+') NEvents=' + str(NTotalEvents/1000)+'K" ' + mail)
	exit(0);

name = "Run_"+str(firstRun)+"_to_"+str(lastRun)
if len(calMode)>0:  name = name+"_"+calMode
if(usePCL==True):   name = name+"_PCL"
else:               name = name+"_CalibTree"
print(name)

oldDirectory = "7TeVData"
newDirectory = "Data_"+name;
os.system("mkdir -p " + newDirectory);
os.system("cp " + oldDirectory + "/* " + newDirectory+"/.");
file = open(newDirectory+"/FileList_cfg.py", "w")
file.write("import FWCore.ParameterSet.Config as cms\n")
file.write("calibTreeList = cms.untracked.vstring()\n")
file.write("#TotalNumberOfEvent considered is %i\n" % NTotalEvents)
file.write(FileList)
file.close()
os.system("cat " + newDirectory + "/FileList_cfg.py")
os.system("sed -i 's|XXX_FIRSTRUN_XXX|"+str(firstRun)+"|g' "+newDirectory+"/*_cfg.py")
os.system("sed -i 's|XXX_LASTRUN_XXX|"+str(lastRun)+"|g' "+newDirectory+"/*_cfg.py")
os.system("sed -i 's|XXX_GT_XXX|"+globaltag+"|g' "+newDirectory+"/*_cfg.py")
os.system("sed -i 's|XXX_PCL_XXX|"+str(usePCL)+"|g' "+newDirectory+"/*_cfg.py")
os.system("sed -i 's|XXX_CALMODE_XXX|"+calMode+"|g' "+newDirectory+"/*_cfg.py")
os.system("sed -i 's|XXX_DQMDIR_XXX|"+DQM_dir+"|g' "+newDirectory+"/*_cfg.py")
os.chdir(newDirectory);

job = initEnv
job+= "\ncd %s; \npwd; \nls; \npython submitJob.py -f %s -l %s -p %s -P %s"%(os.getcwd(),firstRun,lastRun,usePCL,publish)
job+= " -m %s -s %s -a %s"%(calMode,scriptDir,automatic)

print("*** JOB : ***")
print(job)
print("cwd = %s"%(os.getcwd()))
with open("job.sh","w") as f:
   f.write(job)
os.system("chmod +x job.sh")
submitCMD =  'bsub  -q 2nd -J G2prod -R "type == SLC6_64 && pool > 30000" "job.sh"'
print(submitCMD)
os.system(submitCMD)

#if(os.system("sh sequence.sh \"" + name + "\" \"" + calMode + "\" \"CMS Preliminary  -  Run " + str(firstRun) + " to " + str(lastRun) + "\"")!=0):
#	os.system('echo "Gain calibration failed" | mail -s "Gain calibration failed ('+name+')" ' + mail)        
#else:
#	if(publish==True):os.system("sh sequence.sh " + name);
#	os.system('echo "Gain calibration done\nhttps://test-stripcalibvalidation.web.cern.ch/test-stripcalibvalidation/CalibrationValidation/ParticleGain/" | mail -s "Gain calibration done ('+name+')" ' + mail)
#
#if(False and usePCL==True):
#   #Make the same results using the calibTrees for comparisons
#   os.chdir(scriptDir); #go back to initial location
#   os.system('python automatic_RunOnCalibTree.py --firstRun ' + str(firstRun) + ' --lastRun ' + str(lastRun) + ' --publish False --pcl False')
#
#if(automatic==True):
#   #call the script one more time to make sure that we do not have a new run to process
#   os.chdir(scriptDir); #go back to initial location
#   os.system('python automatic_RunOnCalibTree.py')
#
