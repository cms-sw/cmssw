#! /usr/bin/env python
#G.Benelli Aug 26 2010
#Process the CheckAllIOVs.py DEBUG output log files to prepare them as inputs to tkmapcreator...
#Incorporating also the next step: cloning the cfg.py files (1 per log file, i.e. 1 per IOV) and execute cmsRun
#producing the wanted tkMaps.

#TODO:
#Could use this same script to do other processing as input to the ValidateO2O.py for other type of plots etc...

#Script assumes that it is run in the same dir as the output of CheckAllIOVs.py
#TODO:
#In the future can put it in scripts/ and take dir to run from and write to as options.

import os, subprocess

def ProduceTkMapVoltageInputFiles(workdir=os.getcwd()): #Setting the dir by default to the current working directory...
	"""
	Function that processes the indicated workdir directory (defaults to current working directory) looking for CheckAllIOVs.py
	DetVOffReaderDebug*.log files.
	For each log file (that correspond to 1 IOV) 2 input files (one for LV one for HV status) are created for the TkVoltageMapCreator plugin.
	The function returns the 2 lists of HV and LV files.
	"""
	#Get all the files in the directory workdir (1 file per IOV):
	print "Analysing %s directory"%workdir
	logfilenames=[x for x in os.listdir(workdir) if x.startswith("DetVOffReaderDebug")]
	if logfilenames:
		print "Processing %s logfiles..."%len(logfilenames)
	else:
		print "No DetVIfReaderDebug files found!\nPlease run this script in the same dir you have run CheckAllIOVS.py, or in the dir where you store the output logfiles."
	
	#Let's dump for each IOV a TkVoltageMapCreator ready input file, i.e. with 1 entry per detID...
	#Need to read in our usual pkl to get all the strip detids
	#Get it from the release
	import pickle
	#FIX:
	#Until the tag is in the release, CMSSW_RELEASE_BASE should be replaced by CMSSW_BASE...
	StripDetIDAliasPickle=os.path.join(os.getenv("CMSSW_RELEASE_BASE"),"src/CalibTracker/SiStripDCS/data","StripDetIDAlias.pkl")
	#TODO:
	#Could use try/except to make this robust... but from next CVS commit should be fine...
	#Otherwise can use the file in ~gbenelli/O2O/StripDetIDAlias.pkl, use full AFS path!
	DetIDdict=open(StripDetIDAliasPickle,"r")
	StripDetIDAlias=pickle.load(DetIDdict)
	
	#Process the logfiles!
	#Lists to keep the LV/HV input files, that will be returned by the function:
	LVFilenames=[]
	HVFilenames=[]
	for logfilename in logfilenames:
		print logfilename
		#Create LV/HV filenames for the input files we will write
		#TODO:
		#Could add here the possibility of writing in a different dir, by adding an extra outdir argument to the function
		#and use that instead of workdir...
		LVfilename=os.path.join(workdir,logfilename.replace("DetVOffReaderDebug_","LV"))
		HVfilename=os.path.join(workdir,logfilename.replace("DetVOffReaderDebug_","HV"))

		#Adding the filenames to the respective lists that will be returned by the function:
		LVFilenames.append(LVfilename)
		HVFilenames.append(HVfilename)
		
		#Open input/output files:
		logfile=open(logfilename,"r")
		LVfile=open(LVfilename,"w")
		HVfile=open(HVfilename,"w")
		
		#File parsing and creation of 2 separate LV and HV status files:
		#Need dicts to handle the filling of files with all detIDs:
		LVDict={}
		HVDict={}
		for line in logfile:
		    if "OFF" in line: #All the interesting lines contain "OFF" by definition 
		        (detid,HV,LV)=line.split()
			LVDict.update({detid:LV})
			HVDict.update({detid:HV})
	
		#Handle the LV/HV files:
		for detid in StripDetIDAlias.keys():
			detid=str(detid)
			if detid in LVDict.keys(): #Set the detids to whatever status they were reported (can be ON or OFF, since the detid would be reported in the case of HV OFF and LV ON of course...)
				LVfile.write(detid+"\t"+LVDict[detid]+"\n")
			else: #Set the remaining detids as ON (they would be reported otherwise in the LVDict)
				LVfile.write(detid+"\tON\n")
			if detid in HVDict.keys(): #Set the detids to whatever status they were reported (should only reported when OFF..., HV ON while LV OFF should be impossible)
				HVfile.write(detid+"\t"+HVDict[detid]+"\n")
			else: #Set the remaining detids as ON (they would be reported otherwise in the HVDict)
				HVfile.write(detid+"\tON\n")
		
		        
		#Close files:
		logfile.close()
		LVfile.close()
		HVfile.close()
		
	#Now finally return the 2 lists of LV and HV TkVoltageMapCreator input files:
	return LVFilenames,HVFilenames

#Function to run a (for us cmsRun) command:
def runcmd(command):
	"""
	Function that uses subprocess.Popen to run commands, it returns the exit code of the command. 
	"""
	try:
	    process  = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
	    pid=process.pid
	    exitstat= process.wait()
	    cmdout   = process.stdout.read()
	    exitstat = process.returncode
	except OSError as detail:
	    print "Race condition in subprocess.Popen has robbed us of the exit code of the %s process (PID %s).Assume it failed!\n %s\n"%(command,pid,detail)
	    exitstat=999
	if exitstat == None:
	    print "Something strange is going on! Exit code was None for command %s: check if it really ran!"%command
	    exitstat=0
	return exitstat

def CreateTkVoltageMapsCfgs(workdir=os.getcwd()): #Default to current working directory (could pass HV/LV list)
	"""
	Function that looks for TkVoltageMapCreator input files, creates 1 cfg.py for each IOV.
	It returns the list of cfgs ready to be cmsRun to produce the maps 
	"""
	#Use HV log files to loop... could use also LV logs...
	HVLogs=[x for x in os.listdir(workdir) if x.startswith("HV") and "FROM" in x and x.endswith(".log")]
	
	#Open the file to use as template
	TkMapCreatorTemplateFile=open(os.path.join(os.getenv("CMSSW_BASE"),"src/CalibTracker/SiStripDCS/test","TkVoltageMapCreator_cfg.py"),"r")
	TkMapCreatorTemplateContent=TkMapCreatorTemplateFile.readlines()
	#Let's do it!
	TkMapCfgFilenames=[]
	for HVlog in HVLogs:
		#Use full path since workdir could be different than current working dir:
		HVlog=os.path.join(workdir,HVlog)
	        #Check if the corresponding LV log is there!
		LVlog=os.path.join(workdir,HVlog.replace("HV","LV"))
		if not os.path.exists(LVlog):
			print "ARGH! Missing LV file for file %s"%HVlog
			print "Will not process the HV file either!"
		TkMapCfgFilename=os.path.join(workdir,HVlog.replace("HV","TkVoltageMap").replace(".log","_cfg.py"))
		TkMapCfgFilenames.append(TkMapCfgFilename)
		TkMapCfg=open(TkMapCfgFilename,"w")
		for line in TkMapCreatorTemplateContent:
			if "LVStatusFile" in line and "#" not in line:
				line='\tLVStatusFile = cms.string("%s"),\n'%LVlog
			if "LVTkMapName" in line and "#" not in line:
				line='\tLVTkMapName = cms.string("%s"),\n'%LVlog.replace(".log",".png")
			if "HVStatusFile" in line and "#" not in line:
				line='\tHVStatusFile = cms.string("%s"),\n'%HVlog
			if "HVTkMapName" in line and "#" not in line:
				line='\tHVTkMapName = cms.string("%s")\n'%HVlog.replace(".log",".png")
			TkMapCfg.write(line)
		TkMapCfg.close()
	        
	TkMapCreatorTemplateFile.close()
	return TkMapCfgFilenames

def CreateTkVoltageMaps(workdir=os.getcwd()): #Default to current working directory for now...
	"""
	Function that looks for TkVoltageMap*cfg.py in the workdir directory and launches each of them
	creating 2 TkVoltageMaps per IOV, one for LV and one of HV status (each as a png file). 
	"""
	TkMapCfgs=[x for x in os.listdir(workdir) if x.startswith("TkVoltageMap") and "FROM" in x and x.endswith("cfg.py")]
	for TkMapCfg in TkMapCfgs:
		#Make sure we run the cfg in the workdir and also the logfile is saved there...
		TkMapCfg=os.path.join(workdir,TkMapCfg)
		cmsRunCmd="cmsRun %s >& %s"%(TkMapCfg,TkMapCfg.replace(".py",".log"))
		print cmsRunCmd
		exitstat=runcmd(cmsRunCmd)
		if exitstat != 0:
			print "Uh-Oh!"
			print "Command %s FAILED!"%cmsRunCmd
			
#Could put in a def main...

#Create the TkVoltageMapCreator input files in the test dir below:
(LVfiles,HVfiles)=ProduceTkMapVoltageInputFiles()
#print LVfiles
#print HVfiles

#Create the actual TkVoltageMaps!
TkMapCfgFilenames=CreateTkVoltageMapsCfgs()
print TkMapCfgFilenames
CreateTkVoltageMaps()

#Finish this up, so that it can be committed, but above all use it!
#Check how to use CheckAllIOVs.py to access the Offline DB directly! Maybe can fold this in?



