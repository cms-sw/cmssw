#!/usr/bin/env python
import os, re
import math, time
import sys

print ('START')

########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
NumberOfJobs=2
ScriptName = "runMuonOverlapPatternGenerator_SF.py" # script to be used with cmsRun
#FileList = "list_dummy.txt" 
#FileList = "list_MuonGunSample_Pt1to1000_106X.txt" # list with all the file directories
FileList = "list_SingleMu_OneOverPt_And_FlatPt1to1000_125X.txt" 
queue = "testmatch" # give bsub queue -- 8nm (8 minutes), 1nh (1 hour), 8nh, 1nd (1day), 2nd, 1nw (1 week), 2nw 
OutputDir = "/eos/cms/store/user/folguera/L1TMuon/OMTF/2023_06_UsePhase2DTs_NewPats/PatternGen_displ/"
generatePatterns = "generatePatterns=True "
########   customization end   #########

path = os.getcwd()
print ('do not worry about folder creation:')
os.system("rm -rf tmp")
os.system("rm -rf exec")
os.system("rm -rf batchlogs")
os.system("mkdir tmp")
os.system("mkdir exec")
os.system("mkdir %s/%s" %(OutputDir,int(time.time())))
print()

##### loop for creating and sending jobs #####
for x in range(1, int(NumberOfJobs)+1):
    ##### creates jobs #######
    outputname = OutputDir
    if x==1: 
        usePhase2DTs = "usePhase2DTs=True "
        outputname = outputname + "Patterns_layerStat_t12_Phase2DTs.xml"
    else:
        usePhase2DTs = "usePhase2DTs=False "    
        outputname = outputname + "Patterns_layerStat_t12.xml"    
    
    with open('exec/job_'+str(x)+'.sh', 'w') as fout:
        fout.write("#!/bin/sh\n")
        fout.write("echo\n")
        fout.write("echo\n")
        fout.write("echo 'START---------------'\n")
        fout.write("echo 'WORKDIR ' ${PWD}\n")
        fout.write("cd "+str(path)+"\n")
        fout.write("eval `scramv1 runtime -sh`\n")
        fout.write("export X509_USER_PROXY=$1\n")
        fout.write("voms-proxy-info -all\n")
        fout.write("voms-proxy-info -all -file $1\n")
        fout.write("cmsRun "+ScriptName+" outputPatternsXMLFile='"+outputname+"' inputFiles_clear inputFiles_load='"+FileList+"' "+generatePatterns+usePhase2DTs+"\n")
        fout.write("echo 'STOP---------------'\n")
        fout.write("echo\n")
        fout.write("echo\n")
    os.system("chmod 755 exec/job_"+str(x)+".sh")


###### create submit.sub file ####
    
os.mkdir("batchlogs")
with open('submit.sub', 'w') as fout:
    fout.write("executable              = $(filename)\n")
    fout.write("arguments               = $(ClusterId)$(ProcId)\n")
    fout.write("output                  = batchlogs/$(ClusterId).$(ProcId).out\n")
    fout.write("error                   = batchlogs/$(ClusterId).$(ProcId).err\n")
    fout.write("log                     = batchlogs/$(ClusterId).log\n")
    fout.write('+JobFlavour = "%s"\n' %(queue))
    fout.write("\n")
    fout.write('Proxy_path=/afs/cern.ch/user/f/folguera/workdir/proxy/x509up_u50826\n')
    fout.write('arguments               = $(Proxy_path) arg2 arg3 arg4\n')
    fout.write("\n")
    fout.write("queue filename matching (exec/job_*sh)\n")
    
###### sends bjobs ######
os.system("echo submit.sub")
os.system("condor_submit submit.sub")
   
print()
print( "your jobs:")
os.system("condor_q")
print()
print()
