#!/usr/bin/env python

import os,sys
import getopt
import commands
import time
import ROOT
import urllib
import string
import optparse

def numberOfEvents(file,mode):
	rootfile = ROOT.TFile.Open(file,'read')
	tree = ROOT.TTree()
	rootfile.GetObject("gainCalibrationTree%s/tree"%mode,tree)
        NEntries = tree.GetEntries()
        rootfile.Close()
        print file +' --> '+str(NEntries)
	return NEntries	


PCLDATASET = '/StreamExpress/Run2015C-PromptCalibProdSiStripGains-Express-v1/ALCAPROMPT' #used if usePCL==True
CALIBTREEPATH = '/store/group/dpg_tracker_strip/comm_tracker/Strip/Calibration/calibrationtree/GR15' #used if usePCL==False
#CALIBTREEPATH = '/store/caf/user/dimattia/CALIB'
#CALIBTREEPATH = "/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12"
#CALIBTREEPATH = "/castor/cern.ch/user/m/mgalanti/calibrationtree/GR11"

runsToVeto = [247388, 247389, 247395247395, 247397, 247982, 248026, 248031, 248033]


#read arguments to the command line
#configure
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-f', '--firstRun'   ,    dest='firstRun'           , help='first run to process (-1 --> automatic)'  , default='-1')
parser.add_option('-l', '--lastRun'    ,    dest='lastRun'            , help='last run to process (-1 --> automatic)'   , default='-1')
parser.add_option('-P', '--publish'    ,    dest='publish'            , help='publish the results'                      , default='True')
parser.add_option('-p', '--pcl'        ,    dest='usePCL'             , help='use PCL output instead of calibTree'      , default='True')
parser.add_option('-m', '--mode'       ,    dest='calMode'            , help='select the statistics type'      , default='')
(opt, args) = parser.parse_args()

scriptDir = os.getcwd()
globaltag = "auto:run2_data" # "GR_E_V49" #"GR_P_V53" V56
firstRun = int(opt.firstRun)
lastRun  = int(opt.lastRun)
calMode  = str(opt.calMode)
MC=""
publish = (opt.publish=='True')
mail = "dimattia@cern.ch"
automatic = True;
usePCL = (opt.usePCL=='True')
maxNEvents = 2000000

if(firstRun!=-1 or lastRun!=-1): automatic = False


DQM_dir = "AlCaReco/SiStripGains" if "AagBunch" not in opt.calMode else "AlCaReco/SiStripGainsAfterAbortGap"

print "firstRun = " +str(firstRun)
print "lastRun  = " +str(lastRun)
print "publish  = " +str(publish)
print "usePCL   = " +str(usePCL)
print "calMode  = " + calMode
print "DQM_dir  = " + DQM_dir


#go to parent directory = test directory
os.chdir("..");

#identify last run of the previous calibration
if(firstRun<=0):
   out = commands.getstatusoutput("ls /afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/ParticleGain/ | grep Run_ | tail -n 1");
   firstRun = int(out[1].split('_')[3])+1
   print "firstRun = " +str(firstRun)


initEnv='cd ' + os.getcwd() + ';'
initEnv+='source /afs/cern.ch/cms/cmsset_default.sh' + ';'
initEnv+='eval `scramv1 runtime -sh`' + ';'


#Get List of Files to process:
NTotalEvents = 0;
run = 0
FileList = ""

if(usePCL==True):
   print("Get the list of PCL output files from DAS")
   results = commands.getstatusoutput(initEnv+"das_client.py  --limit=9999 --query='run dataset="+PCLDATASET+"'")[1].splitlines()
   results.sort()
   for line in results:
      if(line.startswith('Showing')):continue
      if(len(line)<=0):continue
      linesplit = line.split('   ')
      print linesplit
      run     = int(line.split('   ')[0])      
      if(run<firstRun or run in runsToVeto):continue
      if(lastRun>0 and run>lastRun):continue      
      #check that this run at least contains some events
      print("Checking number of events in run %i" % run)
      NEventsDasOut = commands.getstatusoutput(initEnv+"das_client.py  --limit=9999 --query='summary dataset="+PCLDATASET+" run="+str(run)+" | grep summary.nevents'")[1].splitlines()[-1]
      if(not NEventsDasOut.isdigit() ):
         print ("issue with getting number of events from das, skip this run")
         print NEventsDasOut
         continue
      if(FileList==""):firstRun=run;
      NEvents = int(NEventsDasOut)
      if(NEvents<=3000):continue #only keep runs with at least 3K events
      FileList+="#run=" + str(run) + " -->  NEvents="+str(NEvents/1000).rjust(8)+"K\n"
      resultsFiles = commands.getstatusoutput(initEnv+"das_client.py  --limit=9999 --query='file dataset="+PCLDATASET+" run="+str(run)+"'")
      if(int(resultsFiles[0])!=0 or results[1].find('Error:')>=0):
         print ("issue with getting the list of files from das, skip this run")
         print resultsFiles
         continue
      for f in resultsFiles[1].splitlines():
         if(not f.startswith('/')):continue
         FileList+='calibTreeList.extend(["'+f+'"])\n'
      NTotalEvents += NEvents;
      print("Current number of events to process is " + str(NTotalEvents))
      if(automatic==True and NTotalEvents >= maxNEvents):break;
else:
   print("Get the list of calibTree from castor (eos ls " + CALIBTREEPATH + ")")
   calibTreeInfo = commands.getstatusoutput("/afs/cern.ch/project/eos/installation/0.3.84-aquamarine/bin/eos.select ls -l "+CALIBTREEPATH)[1].split('\n');

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
      NEvents = numberOfEvents("root://eoscms//eos/cms"+CALIBTREEPATH+'/'+info[0],calMode);	
      if(NEvents<=3000):continue #only keep runs with at least 3K events
      if(FileList==""):firstRun=run;
      FileList += 'calibTreeList.extend(["root://eoscms//eos/cms'+CALIBTREEPATH+'/'+info[0]+'"]) #' + str(size).rjust(6)+'MB  NEvents='+str(NEvents/1000).rjust(8)+'K\n'
      NTotalEvents += NEvents;
      print("Current number of events to process is " + str(NTotalEvents))
      if(automatic==True and NTotalEvents >= maxNEvents):break;


if(lastRun<=0):lastRun = run

print "RunRange=[" + str(firstRun) + "," + str(lastRun) + "] --> NEvents=" + str(NTotalEvents/1000)+"K"
if(automatic==True and NTotalEvents<1000000):	#ask at least 1M events to perform the calibration
	print 'Not Enough events to run the calibration'
        os.system('echo "Gain calibration postponed" | mail -s "Gain calibration postponed ('+str(firstRun)+' to '+str(lastRun)+') NEvents=' + str(NTotalEvents/1000)+'K" ' + mail)
	exit(0);

name = "Run_"+str(firstRun)+"_to_"+str(lastRun)
if len(calMode)>0:  name = name+"_"+calMode
if(usePCL==True):   name = name+"_PCL"
else:               name = name+"_CalibTree"

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
if(os.system("sh sequence.sh \"" + name + "\" \"" + calMode + "\" \"CMS Preliminary  -  Run " + str(firstRun) + " to " + str(lastRun) + "\"")!=0):
	os.system('echo "Gain calibration failed" | mail -s "Gain calibration failed ('+name+')" ' + mail)        
else:
	if(publish==True):os.system("sh sequence.sh " + name);
	os.system('echo "Gain calibration done\nhttps://test-stripcalibvalidation.web.cern.ch/test-stripcalibvalidation/CalibrationValidation/ParticleGain/" | mail -s "Gain calibration done ('+name+')" ' + mail)

if(usePCL==True):
   #Make the same results using the calibTrees for comparisons
   os.chdir(scriptDir); #go back to initial location
   os.system('python automatic_RunOnCalibTree.py --firstRun ' + str(firstRun) + ' --lastRun ' + str(lastRun) + ' --publish False --pcl False')

if(automatic==True):
   #call the script one more time to make sure that we do not have a new run to process
   os.chdir(scriptDir); #go back to initial location
   os.system('python automatic_RunOnCalibTree.py')

