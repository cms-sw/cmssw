#!/usr/bin/env python

import os,sys
import getopt
import commands
import time
import ROOT

def numberOfEvents(file):
	rootfile = ROOT.TFile.Open(file,'read')
	tree = ROOT.TTree()
	rootfile.GetObject("commonCalibrationTree/tree",tree)
        NEntries = tree.GetEntries()
        rootfile.Close()
        print file +' --> '+str(NEntries)
	return NEntries	


globaltag = "GR_P_V32" 
path = "/store/group/tracker/strip/calibration/calibrationtree/GR12" #"/castor/cern.ch/user/m/mgalanti/calibrationtree/GR11"
#path = "/castor/cern.ch/user/m/mgalanti/calibrationtree/GR12/"
firstRun = 191919
#firstRun = 192701	#value of the first run with the new calibration --> this is needed to avoid mixing runs with different calibrations
lastRun  = 197000
MC=""
publish = True
mail = "loic.quertenmont@gmail.com"
automatic = True;

#go to parent directory = test directory
os.chdir("..");
#identify last run of the previous calibration
if(firstRun<=0):
	out = commands.getstatusoutput("ls /afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/ParticleGain/ | grep Run_ | tail -n 1");
	firstRun = int(out[1].split('_')[3])+1

#Get List of CalibFiles:
print("Get the list of calibTree from castor (cmsLs" + path + ")")
calibTreeList = ""
calibTreeInfo = commands.getstatusoutput("cmsLs "+path)[1].split('\n');
NTotalEvents = 0;
run = 0
print("Check the number of events available");
for info in calibTreeInfo:
        if(len(info)<1):continue;
	subParts = info.split();        
	size = int(subParts[1])/1048576;
	if(size < 50): continue	#skip file<50MB
	run = int(subParts[4].replace(path+'/',"").replace("calibTree_","").replace(".root","")) 
	if(run<firstRun):continue
        if(lastRun>0 and run>lastRun):continue
	os.system("stager_get -M " + subParts[4] + " &");	
	NEvents = numberOfEvents("root://eoscms//eos/cms"+subParts[4]);	
	if(calibTreeList==""):firstRun=run;
	calibTreeList += '  "root://eoscms//eos/cms'+subParts[4]+'", #' + str(size).rjust(6)+'MB  NEvents='+str(NEvents/1000).rjust(8)+'K\n'
	NTotalEvents += NEvents;
	if(NTotalEvents>2500000):
		break;

if(lastRun==0):lastRun = run

#print calibTreeList

print "RunRange=[" + str(firstRun) + "," + str(lastRun) + "] --> NEvents=" + str(NTotalEvents/1000)+"K"
if(automatic==True and NTotalEvents<1000000):	#ask at least 1M events to perform the calibration
	print 'Not Enough events to run the calibration'
        os.system('echo "Gain calibration postponed" | mail -s "Gain calibration postponed ('+str(firstRun)+' to '+str(lastRun)+') NEvents=' + str(NTotalEvents/1000)+'K" ' + mail)
#	exit(0);

name = "Run_"+str(firstRun)+"_to_"+str(lastRun); 
	
oldDirectory = "7TeVData"
newDirectory = "Data_"+name;
os.system("mkdir -p " + newDirectory);
os.system("cp " + oldDirectory + "/* " + newDirectory+"/.");
os.system("sed -i 's/XXX_CALIBTREE_XXX/"+calibTreeList.replace('\n','\\n').replace('/','\/')+"/g' "+newDirectory+"/*_cfg.py")
os.system("sed -i 's/XXX_FIRSTRUN_XXX/"+str(firstRun)+"/g' "+newDirectory+"/*_cfg.py")
os.system("sed -i 's/XXX_LASTRUN_XXX/"+str(lastRun)+"/g' "+newDirectory+"/*_cfg.py")
os.system("sed -i 's/XXX_GT_XXX/"+globaltag+"/g' "+newDirectory+"/*_cfg.py")
os.chdir(newDirectory);
if(os.system("sh sequence.sh")!=0):
	os.system('echo "Gain calibration failed" | mail -s "Gain calibration failed ('+name+')" ' + mail)
else:
	if(publish==True):os.system("sh sequence.sh " + name);
	os.system('echo "Gain calibration done\nhttps://test-stripcalibvalidation.web.cern.ch/test-stripcalibvalidation/CalibrationValidation/ParticleGain/" | mail -s "Gain calibration done ('+name+')" ' + mail)
