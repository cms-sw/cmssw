#!/usr/bin/env python

import urllib
import string
import os
import sys


Input_ConfigFile = "MergeJob_cfg.py"
                                        # The name of your config file
                                        # where you have to replace the OutputAdresse by XXX_OUTPUT_XXX
                                        # and the Number of Events by XXX_NEVENTS_XXX
                                        # and the Number of Event to skip is XXX_SKIPEVENT_XXX

JobName = "MERGE"              # The name of your job
Output_Path = "."                       # Your OuputPath  (will replace XXX_OUTPUT_XXX)

Job_NEvents = -1                     # Number of Events by job (will replace XXX_NEVENTS_XXX)
Job_Start   = 0                         # The Index of your first job
Job_End     = 1                         # The Index of your last  job
Job_SEvents = 0                         # Event that you want to skip
                                        # The first event will be Job_SEvents+1
Job_Seed = 123456789                    # Seed of the job... Useful for RandomService Initialisation


FarmDirectory = "FARM_Merge"
QUEUE         = "cmscaf"



def CreateTheConfigFile(CONFIG_FILE,NEVENTS,OUTPUTFILE,INDEX):

	config_file=open(CONFIG_FILE,'r')
	config_txt   = config_file.read()               # Read all data
	config_file.close()
	newconfig_path = "./"+FarmDirectory+"/InputFile/%04i_" % INDEX
        newconfig_path = newconfig_path + JobName + "_cfg.py"

	mylogo1 = "#  -----------------------------------------------\n"
	mylogo2 = "# |   cfg modified by the LaunchOnFarm Script     |\n"
	mylogo3 = "# |   Created by Loic Quertenmont                 |\n"
	mylogo4 = "# |   Loic.quertenmont@cern.ch                    |\n"
	mylogo5 = "#  -----------------------------------------------\n\n\n\n"
	config_txt = mylogo1 + mylogo2 + mylogo3 + mylogo4 + mylogo5 + config_txt

        newconfig_file=open(newconfig_path,'w')
        newconfig_file.write("%s" % config_txt)
        newconfig_file.close()


	i=0  
	while i < len(config_txt) :
		if config_txt[i:i+15]=='XXX_NEVENTS_XXX':
			Skip = INDEX*NEVENTS+Job_SEvents
			MaxEvent = NEVENTS
			print("job #%d" %INDEX + "\t\tNumber of Events fixed to \t\t%d"%MaxEvent)
			newconfig_file=open(newconfig_path,'w')
			newconfig_file.write("%s" % config_txt[0:i])
			newconfig_file.write("%d" % MaxEvent)
			newconfig_file.write("%s" % config_txt[i+15:len(config_txt)])
			newconfig_file.close()
			newconfig_file=open(newconfig_path,'r')
			config_txt   = newconfig_file.read()
			newconfig_file.close()
			i = 0
                if config_txt[i:i+12]=='XXX_SEED_XXX':
                        Skip = INDEX*NEVENTS+Job_SEvents
                        seed = Job_Seed + INDEX
                        print("job #%d" %INDEX + "\t\tSeed fixed to \t\t%d"%seed)
                        newconfig_file=open(newconfig_path,'w')
                        newconfig_file.write("%s" % config_txt[0:i])
                        newconfig_file.write("%d" % seed)
                        newconfig_file.write("%s" % config_txt[i+12:len(config_txt)])
                        newconfig_file.close()
                        newconfig_file=open(newconfig_path,'r')
                        config_txt   = newconfig_file.read()
                        newconfig_file.close()
                        i = 0
		if config_txt[i:i+14]=='XXX_OUTPUT_XXX':
			print("job #%d" %INDEX + "\tOutput file fixed to\t\t%s"%OUTPUTFILE)
			newconfig_file=open(newconfig_path,'w')
			newconfig_file.write("%s" % config_txt[0:i])
			newconfig_file.write("%s"% OUTPUTFILE)
			newconfig_file.write("%s" % config_txt[i+14:len(config_txt)])
			newconfig_file.close()
			newconfig_file=open(newconfig_path,'r')
			config_txt   = newconfig_file.read()
			newconfig_file.close()
			i = 0
                if config_txt[i:i+17]=='XXX_SKIPEVENT_XXX':
			Skip = INDEX*NEVENTS+Job_SEvents
                        print("job #%d" %INDEX + "\tNumber of Event to skip is fixed to\t\t%i"%Skip)
                        newconfig_file=open(newconfig_path,'w')
                        newconfig_file.write("%s" % config_txt[0:i])
                        newconfig_file.write("%i"%Skip)
                        newconfig_file.write("%s" % config_txt[i+17:len(config_txt)])
                        newconfig_file.close()
                        newconfig_file=open(newconfig_path,'r')
                        config_txt   = newconfig_file.read()
                        newconfig_file.close()
                        i = 0
                if config_txt[i:i+9]=='XXX_I_XXX':                       
                        newconfig_file=open(newconfig_path,'w')
                        newconfig_file.write("%s" % config_txt[0:i])
                        newconfig_file.write("%04i"%INDEX)
                        newconfig_file.write("%s" % config_txt[i+9:len(config_txt)])
                        newconfig_file.close()
                        newconfig_file=open(newconfig_path,'r')
                        config_txt   = newconfig_file.read()
                        newconfig_file.close()
                        i = 0
                if config_txt[i:i+12]=='XXX_PATH_XXX':
                        newconfig_file=open(newconfig_path,'w')
                        newconfig_file.write("%s" % config_txt[0:i])
                        newconfig_file.write("%s" % path)
                        newconfig_file.write("%s" % config_txt[i+12:len(config_txt)])
                        newconfig_file.close()
                        newconfig_file=open(newconfig_path,'r')
                        config_txt   = newconfig_file.read()
                        newconfig_file.close()
                        i = 0

		i = i+1

def CreateTheShellFile(INITDIR,INDEX):
	shell_path = "./"+FarmDirectory+"/InputFile/%04i_" % INDEX
        shell_path = shell_path + JobName + ".sh"

        cfg_path = INITDIR+"/"+FarmDirectory+"/InputFile/%04i_" % INDEX
        cfg_path = cfg_path + JobName + "_cfg.py"

        TxtIndex = "%04i" % INDEX  

	shell_file=open(shell_path,'w')
	shell_file.write("#! /bin/sh\n")
	shell_file.write("#  ----------------------------------------------- \n")
	shell_file.write("# |   Script created by the LaunchOnFarm Script   |\n")
	shell_file.write("# |   Created by Loic Quertenmont                 |\n")
	shell_file.write("# |   Loic.quertenmont@cern.ch                    |\n")
	shell_file.write("#  ----------------------------------------------- \n\n\n\n")
#	shell_file.write("%s\n" % "export SCRAM_ARCH=slc4_ia32_gcc345")
#        shell_file.write("%s\n" % "export BUILD_ARCH=slc4_ia32_gcc345")
#        shell_file.write("%s\n" % "export VO_CMS_SW_DIR=/nfs/soft/cms")
#	shell_file.write("%s\n" % "source /nfs/soft/cms/cmsset_default.sh")
	shell_file.write("cd " + path + "\n")
	shell_file.write("%s\n" % "eval `scramv1 runtime -sh`")
	shell_file.write("%s\n" % "cd -")
        shell_file.write("%s" % "cmsRun " + cfg_path +"\n\n\n")
        shell_file.write("mv %s* " % JobName)
#        shell_file.write("mv * ")
        shell_file.write("%s/" % path)
        shell_file.write("%s/.\n" % Output_Path)
	shell_file.close()
	chmod_path = "chmod 777 "+shell_path
	os.system(chmod_path)


path = os.getcwd() 	#Get the current path
if os.path.isdir(FarmDirectory) == False:
	os.system('mkdir '+FarmDirectory)
	os.system('mkdir '+FarmDirectory+'/Outputs')
	os.system('mkdir '+FarmDirectory+'/Log')
	os.system('mkdir '+FarmDirectory+'/InputFile')

for i in range(Job_Start,Job_End):
	print('Submitting job number %d' %i)

	CreateTheConfigFile(Input_ConfigFile,Job_NEvents,JobName,i)
	CreateTheShellFile(path,i)

	condor_path = path + "/"+FarmDirectory+"/InputFile/%04i_" % i
        condor_path = condor_path + JobName + ".cmd"

        shell_path = path + "/"+FarmDirectory + "/InputFile/%04i_" % i
        shell_path = shell_path + JobName + ".sh"
   
        cdPath = "cd " + path
        cdFarm = "cd " + path + "/" + FarmDirectory
        os.system(cdFarm)

        JobName = "'" + JobName + "%04i'" % i
        OutputPath = "'out/'"
        batchSubmit = "bsub -q " + QUEUE + " -J " + JobName  + " '" + shell_path + " 0 ele'"
        os.system (batchSubmit)

	print('\n')
NJobs = Job_End - Job_Start
print("\n\n")
print("\t\t\t%i Jobs submitted by the LaunchOnFarm script" % NJobs)
print("\t\t\t         Created by Loic Quertenmont")
print("\t\t\t           Loic.quertenmont@cern.ch")
print("\n\n")
