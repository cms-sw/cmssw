#!/usr/bin/env python
import urllib
import string
import os
import sys
import commands
import time
import optparse
import submitCalibTree.Config
import submitCalibTree.launchJobs

mailAdd = ""
start = time.strftime("%D %H:%M")

def mail(STDruns,AAGruns,cleanUpLog):
   if mailAdd=="":
      print "No email address specified."
      return

   message  = "Production started at %s\n"%start
   message += "             ended at %s\n"%time.strftime("%D %H:%M")
   message += "\n\n\n"
   message += "Std bunch : processed the following runs :\n"
   previousRun=0
   nJobs=1
   runs = {}
   for run in STDruns:
      if not run[0] in runs.keys():
         runs[run[0]]=1
      else:
         runs[run[0]]+=1

   runsOrdered = sorted(runs.keys())

   for r in runsOrdered:
      message+=" Run %s (%s jobs) \n"%(r,runs[r])
   message +="\n\n"
   message += "Aag bunch : processed the following runs :\n"
   runs={}
   for run in AAGruns:
      if not run[0] in runs.keys():
         runs[run[0]]=1
      else:
         runs[run[0]]+=1
   runsOrdered = runs.keys()
   runsOrdered.sort()

   for r in runsOrdered:
      message+=" Run %s (%s jobs) \n"%(r,runs[r])

   message+="\n\n **** Cleaning report **** \n\n"
   message+=cleanUpLog.replace("\"","").replace("'","")
   os.system('echo "%s" | mail -s "CalibTree production status" '%message + mailAdd)

debug = False
for arg in sys.argv:
   debug +="debug" in arg.lower()
if debug:
   print "DEBUG MODE DETECTED"
   print "--> Will not submit jobs"
   print "--> Will not modify status files"
   print



#Std bunch:
print "Processing Std bunch"
config = submitCalibTree.Config.configuration(debug=debug)
cleanUpMessage = "Unable to clean up folder. Bad clean-up integrity?"

if config.integrity:
   print "Cleaning up directory..."
   cleanUpMessage = commands.getstatusoutput("cd %s; python cleanFolders.py; cd -"%config.RUNDIR)[1]
   print cleanUpMessage
   with open("LastRun.txt","r") as lastRun:
      for line in lastRun:
         line = line.replace("\n","").replace("\r","")
         if line.isdigit():
            config.firstRun = int(line)

   with open("FailledRun.txt","r") as failled:
      for line in failled:
         line = line.split()
         if len(line)==1:
            if line[0].isdigit() and len(line[0])==6:
               config.relaunchList.append(line)
         elif len(line)==3:
            if line[0].isdigit() and line[1].isdigit() and line[2].isdigit and len(line[0])==6:
               config.relaunchList.append(line)
   if not debug:
      with open("FailledRun.txt","w") as failled:
         failled.write("")

   lastRunProcessed = submitCalibTree.launchJobs.generateJobs(config)

   print config.launchedRuns

   if not debug:
      with open("LastRun.txt","w") as lastRun:
         lastRun.write(str(lastRunProcessed))



print "Processing AAG"



configAAG = submitCalibTree.Config.configuration(True,debug=debug)
if configAAG.integrity:
   with open("LastRun_Aag.txt","r") as lastRun:
      for line in lastRun:
         line = line.replace("\n","").replace("\r","")
         if line.isdigit():
            configAAG.firstRun = int(line)


   with open("FailledRun_Aag.txt","r") as failled:
      for line in failled:
         line = line.split()
         if len(line)==1:
            if line[0].isdigit() and len(line[0])==6:
               configAAG.relaunchList.append(line)
         elif len(line)==3:
            if line[0].isdigit() and line[1].isdigit() and line[2].isdigit and len(line[0])==6:
               configAAG.relaunchList.append(line)
   if not debug:
      with open("FailledRun_Aag.txt","w") as failled:
         failled.write("")

   lastRunProcessed = submitCalibTree.launchJobs.generateJobs(configAAG)

   if not debug:
      with open("LastRun_Aag.txt","w") as lastRun:
         lastRun.write(str(lastRunProcessed))


mail(config.launchedRuns,configAAG.launchedRuns,cleanUpMessage)
