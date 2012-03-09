#!/usr/bin/env python

import os,sys
import getopt
import commands
import time
import ROOT

def numberOfEvents(file):
	rootfile = ROOT.TFile.Open(file,'read')
	tree = ROOT.TTree()
	rootfile.GetObject("gainCalibrationTree/tree",tree)
	return tree.GetEntries()	


globaltag = "GR_P_V13" 
path = "/castor/cern.ch/user/m/mgalanti/calibrationtree/GR11"
firstRun = 0
#lastRun  = 182890
lastRun  = 0
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
calibTreeList = ""
calibTreeInfo = commands.getstatusoutput("rfdir "+path)[1].split('\n');
NTotalEvents = 0;
run = 0
for info in calibTreeInfo:
	subParts = info.split();
	size = int(subParts[4])/1048576;
	if(size < 50): continue	#skip file<50MB
	run = int(subParts[8].replace("calibTree_","").replace(".root","")) 
	if(run<firstRun):continue
        if(lastRun>0 and run>lastRun):continue
	os.system("stager_get -M " + path+'/'+subParts[8]);	
	NEvents = numberOfEvents("rfio:"+path+"/"+subParts[8]);	
	if(calibTreeList==""):firstRun=run;
	calibTreeList += '  "rfio:'+path+'/'+subParts[8]+'", #' + str(size).rjust(6)+'MB  NEvents='+str(NEvents/1000).rjust(8)+'K\n'
	NTotalEvents += NEvents;
	if(NTotalEvents>2500000):
		break;
lastRun = run

#print calibTreeList

print "RunRange=[" + str(firstRun) + "," + str(lastRun) + "] --> NEvents=" + str(NTotalEvents/1000)+"K"
if(automatic==True and NTotalEvents<2500000):	#ask at least 5M events to perform the calibration
	print 'Not Enough events to run the calibration'
	exit(0);

name = "Run_"+str(firstRun)+"_to_"+str(lastRun); 
	
oldDirectory = "7TeVData"
newDirectory = "testData_"+name;
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
