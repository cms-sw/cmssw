#!/usr/bin/env python

import urllib
import string
import os
import sys

Input_ConfigFile = "MultiJob_cfg.py"
					# The name of your config file
					# where you have to replace the OutputAdresse by XXX_OUTPUT_XXX
					# and the Number of Events by XXX_NEVENTS_XXX
					# and the Number of Event to skip is XXX_SKIPEVENT_XXX
Input_CffFile    = "InputFiles_cff.py"
Input_CffN       = 1


Output_RootFile = "ALCA" # The name of your output file  (will replace XXX_OUTPUT_XXX)

Job_NEvents = -1			# Number of Events by job (will replace XXX_NEVENTS_XXX)
Job_Start   = 0 			# The Index of your first job
Job_End     = 115			# The Index of your last  job
Job_SEvents = 0				# Event that you want to skip
					# The first event will be Job_SEvents+1

FarmDirectory = "FARM"
QUEUE         = "cmscaf"



def CreateTheConfigFile(PATH,CONFIG_FILE,NEVENTS,OUTPUTFILE,INPUTFILE,INDEX):

	config_file=open(CONFIG_FILE,'r')
	config_txt   = config_file.read()               # Read all data
	config_file.close()
	newconfig_path = PATH + "/"+FarmDirectory+"/InputFile/%04i_" % INDEX
        newconfig_path = newconfig_path + Output_RootFile + "_cfg.py"

	mylogo1 = "#  -----------------------------------------------\n"
	mylogo2 = "# |   cfg modified by the LaunchOnFarm Script     |\n"
	mylogo3 = "# |   Created by Loic Quertenmont                 |\n"
	mylogo4 = "# |   Loic.quertenmont@cern.ch                    |\n"
	mylogo5 = "#  -----------------------------------------------\n\n\n\n"
	config_txt = mylogo1 + mylogo2 + mylogo3 + mylogo4 + mylogo5 + config_txt

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
		if config_txt[i:i+14]=='XXX_OUTPUT_XXX':
			print("job #%d" %INDEX + "\tOutput file fixed to\t\t%s"%OUTPUTFILE)
			newconfig_file=open(newconfig_path,'w')
			newconfig_file.write("%s" % config_txt[0:i])
			newconfig_file.write("%s"% OUTPUTFILE)
			newconfig_file.write("_%04i.root" % INDEX)
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
                if config_txt[i:i+13]=='XXX_INPUT_XXX':
                        print("job #%d" %INDEX + "\tInput file fixed to\t\t%s"%INPUTFILE)
                        newconfig_file=open(newconfig_path,'w')
                        newconfig_file.write("%s" % config_txt[0:i])
                        newconfig_file.write("%s" % GetInputFiles(PATH,Input_CffFile,NEVENTS,OUTPUTFILE,INDEX))
                        newconfig_file.write("%s" % config_txt[i+13:len(config_txt)])
                        newconfig_file.close()
                        newconfig_file=open(newconfig_path,'r')
                        config_txt   = newconfig_file.read()
                        newconfig_file.close()
                        i = 0



		i = i+1

def GetInputFiles(PATH,INPUT_FILE,NEVENTS,OUTPUTFILE,INDEX):

        config_file=open(INPUT_FILE,'r')
        config_txt = ""
	i=0
        iMin = (INDEX+0)*Input_CffN
	iMax = (INDEX+1)*Input_CffN-1
        for line in config_file.xreadlines():
                if(line[0:1]!='\''):
			continue

                if( (i>=iMin) and (i<=iMax) ):
			config_txt = config_txt + line
                i = i+1

        if(iMax>=i):
		return 0
        config_file.close()


        if(config_txt[len(config_txt)-2:len(config_txt)-1]==','):
		config_txt = config_txt[0:len(config_txt)-2]
        newconfig_path = PATH + "/"+FarmDirectory+"/InputFile/%04i_" % INDEX
        newconfig_path = newconfig_path + Output_RootFile + "_cff.py"

        return config_txt



def CreateTheShellFile(PATH,INDEX):
	shell_path = "./"+FarmDirectory+"/InputFile/%04i_" % INDEX
        shell_path = shell_path + Output_RootFile + ".sh"

        cfg_path = PATH + "/" + FarmDirectory + "/InputFile/%04i_" % INDEX
        cfg_path = cfg_path + Output_RootFile + "_cfg.py"

	shell_file=open(shell_path,'w')
	shell_file.write("#! /bin/sh\n")
	shell_file.write("#  ----------------------------------------------- \n")
	shell_file.write("# |   Script created by the LaunchOnFarm Script   |\n")
	shell_file.write("# |   Created by Loic Quertenmont                 |\n")
	shell_file.write("# |   Loic.quertenmont@cern.ch                    |\n")
	shell_file.write("#  ----------------------------------------------- \n\n\n\n")
        shell_file.write("%s" % "cd " + PATH + "/" + FarmDirectory + "\n")
	shell_file.write("%s\n" % "eval `scramv1 runtime -sh`")
#        shell_file.write("%s\n" % "export STAGE_SVCCLASS=cmscaf")
#        shell_file.write("%s\n" % "export STAGER_TRACE=3")
        shell_file.write("%s" % "cmsRun " + cfg_path +"\n")
	shell_file.close()
	chmod_path = "chmod 777 "+shell_path
	os.system(chmod_path)


path = os.getcwd() 	#Get the current path
os.system('mkdir '+FarmDirectory)
os.system('mkdir '+FarmDirectory+'/RootFiles')
os.system('mkdir '+FarmDirectory+'/Log')
os.system('mkdir '+FarmDirectory+'/InputFile')

for i in range(Job_Start,Job_End):
	print('Submitting job number %d' %i)

        input_path = FarmDirectory + ".InputFile.%04i_" % i
        input_path = input_path + Output_RootFile + "_cff.py"


#        if( GetInputFiles(path,Input_CffFile,Job_NEvents,path+"/"+FarmDirectory+"/RootFiles/"+Output_RootFile,i) == 0)
#               print('error during the _cfg.py file creation --> are you sure InputFile_cff.py contains enough lines? \n')
#               continue
		
        
#	cff_created = 0
#        if(len(Input_CffFile)>3):
#		cff_created = CreateTheInputFile(path,Input_CffFile,Job_NEvents,path+"/"+FarmDirectory+"/RootFiles/"+Output_RootFile,i)
#        if(cff_created==0):
#                print('error during the cff file creation --> are you sure it contains enough lines? \n')
#		continue
	CreateTheConfigFile(path,Input_ConfigFile,Job_NEvents,path+"/"+FarmDirectory+"/RootFiles/"+Output_RootFile,input_path,i)
	CreateTheShellFile(path,i)

	condor_path = "./"+FarmDirectory+"/InputFile/%04i_" % i
        condor_path = condor_path + Output_RootFile + ".cmd"

        shell_path = path + "/" + FarmDirectory + "/InputFile/%04i_" % i
        shell_path = shell_path + Output_RootFile + ".sh"
         
        JobName = "'" + Output_RootFile + "%04i'" % i
#        OutputPath = "Log/%04i_" % i
#        OutputPath = OutputPath + Output_RootFile + "/"
        OutputPath = "'out/'"

#        batchSubmit = "bsub -q" + QUEUE + " -J" + JobName  + "'" + shell_path + " 0 ele'"
#        batchSubmit = "bsub -q " + QUEUE + " -J " + JobName  + " -oo " + OutputPath + " -eo " + OutputPath + " '" + shell_path + " 0 ele'"
        batchSubmit = "bsub -q " + QUEUE + " -J " + JobName  + " '" + shell_path + " 0 ele'"
	os.system (batchSubmit)
        
	print('\n')
NJobs = Job_End - Job_Start
print("\n\n")
print("\t\t\t%i Jobs submitted by the LaunchOnFarm script" % NJobs)
print("\t\t\t         Created by Loic Quertenmont")
print("\t\t\t           Loic.quertenmont@cern.ch")
print("\n\n")
