#!/usr/bin/env python
import urllib
import Config
import string
import os
import sys
import commands
import time
import optparse

def checkDatasetStructure(dataset,silent=False):
   goodDataset=True
   d = dataset.split("/")
   if not len(d) == 4:
      if not silent:
         print "Bad dataset. Expecting 4 '/'"
      goodDataset=False
      return False
   if not d[0]=='':
      if not silent:
         print "Bad dataset. Expecting nothing before first '/'"
      goodDataset=False
   if not len(d[1])>0 or not len(d[2]) > 0 or not len(d[3]) > 0:
      if not silent:
         print "Bad dataset. Expecting text between '/'"
      goodDataset=False
   if os.path.isdir(dataset):
      if not silent:
         print "Bad dataset. Can't be an existing directory"
      goodDataset=False
   return goodDataset

def getDatasetFromPattern(pattern,conf):
   if not checkDatasetStructure(pattern):
      print "FATAL ERROR, bad dataset pattern"
      return([])
   cmd = "das_client  --limit=9999 --query='dataset dataset="+pattern+"'"
   print cmd

   result = commands.getstatusoutput(conf.initEnv+cmd)[1].splitlines()
   print conf.initEnv+cmd 
   datasets = []
   for line in result:
      print line
      if checkDatasetStructure(line,silent=False):
         datasets.append(line)
   return datasets

def getRunsFromDataset(dataset,conf):
   if not checkDatasetStructure(dataset):
      print "FATAL ERROR, bad dataset pattern"
      return([])
   cmd = "das_client  --limit=9999 --query='run dataset="+dataset+"'"
   print cmd
   result = commands.getstatusoutput(conf.initEnv+cmd)[1].splitlines()
   
   runs=[]
   for line in result:
      if line.isdigit:
         if len(line)==6: #We want the run number to be btw 100.000 and 999.999
            runs.append(int(line))
   runs.sort() 
   return runs

def getNumberOfEvents(run,dataset,conf):
   if not int(run) > 99999 or not int(run)<1000000:
      print "Invalid run number"
      return 0
   if not checkDatasetStructure(dataset):
      print "Invalid dataset"
      return 0
   NEventsDasOut = commands.getstatusoutput(conf.initEnv+"das_client  --limit=9999 --query='summary dataset="+dataset+" run="+str(run)+" | grep summary.nevents'")[1].    splitlines()[-1]
   if not NEventsDasOut.isdigit():
      print "Invalid number of events"
      return 0
   else:
      return int(NEventsDasOut)

def getNumberOfFiles (run,dataset,conf):
   if not int(run) > 99999 or not int(run)<1000000:
      print "Invalid run number"
      return 0
   if not checkDatasetStructure(dataset):
      print "Invalid dataset"
      return 0
   NFilesDasOut = commands.getstatusoutput(conf.initEnv+"das_client  --limit=9999 --query='summary dataset="+dataset+" run="+str(run)+" | grep summary.nfiles'")[1].      splitlines()[-1]
   if not NFilesDasOut.isdigit():
      print "Invalid number of files."
      return 0
   else :
      return int(NFilesDasOut)


def reSubmitJob(run, dataset, conf, first, last):
   print "Re-submitting jobs for run = %s, dataset = %s"%(run, dataset)

   #GET THE LIST OF FILE FROM THE DATABASE
   files = ''
   if not checkDatasetStructure(dataset,conf):
      print "FATAL ERROR, bad dataset"
      return -1
   if not run > 99999 or not run<1000000:
      print "FATAL ERROR, bad run number"
      return -1
   results = commands.getstatusoutput(conf.initEnv+"das_client  --limit=9999 --query='file dataset="+dataset+" run="+str(run)+"'")
   if(int(results[0])!=0 or results[1].find('Error:')>=0):
      print "FATAL ERROR, unable to get files list"
      print results
      #os.system('echo ' + str(run) + ' >> FailledRun%s.txt'%('_Aag' if AAG else ''))
      return -1
   filesList = results[1].splitlines();
   fileId = 0
   for f in filesList:
      if(not f.startswith('/store')):continue
      if fileId>=int(first) :
         files+="'"+f+"',"
      fileId+=1
      if fileId>int(last) :      
         sendJob(dataset,run,files,conf,first)
         return (1)


def submitJobs(run, dataset, nFiles, conf):
   print "Submitting jobs for run = %s, dataset = %s"%(run, dataset)
   
   #GET THE LIST OF FILE FROM THE DATABASE
   files = ''
   if not checkDatasetStructure(dataset,conf):
      print "FATAL ERROR, bad dataset"
      return -1
   if not run > 99999 or not run<1000000:
      print "FATAL ERROR, bad run number"
      return -1
   results = commands.getstatusoutput(conf.initEnv+"das_client  --limit=9999 --query='file dataset="+dataset+" run="+str(run)+"'")
   if(int(results[0])!=0 or results[1].find('Error:')>=0):
      print "FATAL ERROR, unable to get files list"
      print results
      #os.system('echo ' + str(run) + ' >> FailledRun%s.txt'%('_Aag' if AAG else ''))
      return -1
   filesList = results[1].splitlines();
   filesInJob = 0
   firstFile = 0
   for f in filesList:
      if(not f.startswith('/store')):continue
      if filesInJob<conf.nFilesPerJob: 
         files+="'"+f+"',"
         filesInJob+=1
      else:
         firstFile = firstFile+filesInJob
         sendJob(dataset,run,files,conf,firstFile) 
         files="'"+f+"',"
         filesInJob=1
   sendJob(dataset,run,files,conf,firstFile)
   
def sendJob(dataset,run,files,conf,firstFile):
   cmd = "python %s/submitCalibTree/runJob.py -f %s --firstFile %s -d %s -r %s "%(conf.RUNDIR, files,firstFile,dataset,run)
   if conf.AAG:
      cmd+=" -a "
   bsub = 'bsub -q 2nd -J calibTree_' + str(run) + '_' + str(firstFile)+ '_' + '_%s'%("Aag" if conf.AAG else 'Std')+' -R "type == SLC6_64 && pool > 30000" ' + ' "'+cmd+'"'
   conf.launchedRuns.append([run,firstFile])
   if conf.submit:
      os.system(bsub)
   else:
      print bsub

def generateJobs(conf):
   print "Gathering jobs to launch."
   print conf
   lastRunProcessed = conf.firstRun
   datasets = getDatasetFromPattern(conf.datasetPat,conf)
   for d in datasets:
      datasetRuns = getRunsFromDataset(d,conf)
      print datasetRuns
      for r in datasetRuns:
         if int(r) > conf.firstRun and int(r)<conf.lastRun:
            print "Checking run %s"%r
            n=getNumberOfEvents(r,d,conf)
            if n < 250:
               print "Skipped. (%s evt)"%n
            else:
               nFiles = getNumberOfFiles(r,d,conf)
               if nFiles > 0:
                  print "Will be processed ! (%s evt, %s files)"%(n,nFiles)
                  if r > lastRunProcessed:
                     lastRunProcessed = r
                  submitJobs(r,d,nFiles,conf)
               else:
                  print "Skipped. (%s evt,%s files)"%(n,nFiles)
         else:
            for failled in conf.relaunchList:
               if int(failled[0]) == int(r):
                  print "Relaunching job %s "% failled
                  if len(failled)==3:
                     reSubmitJob(int(failled[0]),d,conf,failled[1],failled[2])
                  else:
                     submitJobs(int(failled[0]),d,25,conf)
   return lastRunProcessed


def cleanUp():
   os.system('rm core.*')


def checkCorrupted(lastGood, config):
   calibTreeList = ""
   print("Get the list of calibTree from" + config.CASTORDIR + ")")
   calibTreeInfo = commands.getstatusoutput(config.eosLs +config.CASTORDIR)[1].split('\n');
   NTotalEvents = 0;
   runList = []

   for info in calibTreeInfo:
      subParts = info.split();
      if(len(subParts)<4):  continue

      runList.append(subParts[-1].replace("calibTree_","").replace(".root","").split("_"))
   print runList
   datasets = getDatasetFromPattern(config.datasetPat,config)
   for d in datasets:
      datasetRuns = getRunsFromDataset(d,config)
      print datasetRuns
      for r in datasetRuns:
         if int(r) > lastGood:
            print "Checking run %s"%r
            n=getNumberOfEvents(r,d,config)
            if n < 250:
               print "Skipped. (%s evt)"%n
            else:
               nFiles = getNumberOfFiles(r,d,config)
               if nFiles < 25:
                  print "Found run %s ? %s"%(r,[str(r)] in runList) 
               else:
                  x=25
                  while x<nFiles:
                     print "Found run %s , %s ? %s "%(r,x, [str(r),str(x)] in runList)
                     x+=25


               


   #for line in runList:
   #   print line
      #file = "root://eoscms/"+config.CASTORDIR+"/"+subParts[-1]
      #print("Checking " + file)
      #results = commands.getstatusoutput(config.initEnv+'root -l -b -q ' + file)
      #if(len(results[1].splitlines())>3):
      #   print(results[1]);
      #   print("add " + str(run) + " to the list of failled runs")
#         os.system('echo ' + str(run) + ' >> FailledRun%s.txt'%('_Aag' if AAG else ''))
   


if __name__ == "__main__":
   print "DEBUG DEBUG DEBUG"
   c = Config.configuration(False)
   c.runNumber = 1
   c.firstRun  = 274500
   c.lastRun   = 275000
   c.debug     = True
   #generateJobs(c)
   checkCorrupted(0,c)

if False:
#elif(checkCorrupted):
   #### FIND ALL CORRUPTED FILES ON CASTOR AND MARK THEM AS FAILLED RUN

   calibTreeList = ""
   print("Get the list of calibTree from" + CASTORDIR + ")")
   calibTreeInfo = commands.getstatusoutput(initEnv+"eos ls " + CASTORDIR)[1].split('\n');
   NTotalEvents = 0;
   run = 0
   for info in calibTreeInfo:
      subParts = info.split();
      if(len(subParts)<4):continue
       
      run = int(subParts[4].replace("/calibTree_","").replace(".root","").replace(CASTORDIR,""))
      file = "root://eoscms//eos/cms"+subParts[4] 
      print("Checking " + file)
      results = commands.getstatusoutput(initEnv+'root -l -b -q ' + file)
      if(len(results[1].splitlines())>3):
         print(results[1]);
         print("add " + str(run) + " to the list of failled runs")
         os.system('echo ' + str(run) + ' >> FailledRun%s.txt'%('_Aag' if AAG else ''))
   
   #### If mode = All, relaunch with mode = Aag
   if opt.datasetType.lower()=="all":
      system("cd "+RUNDIR+"; python SubmitJobs.py -c -d Aag")

#else:
   #### UNKNOWN CASE
#   print "unknown argument: make sure you know what you are doing?"
