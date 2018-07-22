from __future__ import print_function
import os

def getName(log):
   if len(log)<10:
      return ("Unable to get name (log too short)")
   else:
      Aag = None
      for line in log:
         if "/store/express/Run2017" in line:
            Aag = "AfterAbortGap" in line
      if Aag == None:
         return ("Unable to get name (no AAG info found)")
      runId=""
      for line in log[15:]:
         if "Processing files" in line:
            runId = line.replace("Processing files ","").replace("to","").replace("of run","").split()
            break
      if runId=="":
         return("Unable to get name (run numbers not found)")
      return("%s %s %s AAG = %s"%(runId[2],runId[0],runId[1],Aag))
      #return log[4][log[9].find("calibTree"):].replace(".root","").replace("_"," ")
def checkRelaunch(log):
   log = log.split(" AAG = ")
   if log[1]=="False":
      relaunchFile = "FailledRun.txt"
   elif log[1]=="True":
      relaunchFile = "FailledRun_Aag.txt"
   else:
      return (-1)
   
   relaunched = 0
   n=0
   with open(relaunchFile,"r") as f:
      for line in f:
         n+=1
         if log[0] in line:
            relaunched = 1
            break
   return(relaunched)
      
def relaunchShort(folder):
   cmd=""
   if not "LSFJOB" in os.listdir(folder):
      print("Unable to open command file")
      return -1
   with open(folder+"/LSFJOB","r") as f:
      for line in f:
         if "job_starter" in line:
            cmd=line.split("job_starter")[1]
            break
   if cmd=="":
      print("Unable to get command...")
      return -1
   AAG       = "SiStripCalMinBiasAfterAbortGap" in cmd
   cmd=cmd.split(" ")
   for i in range(len(cmd)):
      if cmd[i]=="--firstFile":
         firstFile = cmd[i+1].replace("'",'').replace("\n","")
      if cmd[i]=="-r":
         run = cmd[i+1].replace("'",'').replace("\n","")
      if cmd[i]=="-f":
         nRuns = len(cmd[i+1].split(","))-1
   print("Found run specs : %s %s %s (AAG : %s)"%(run,firstFile,int(firstFile)+int(nRuns),AAG))
   name = "%s %s %s AAG = %s"%(run,firstFile,int(firstFile)+int(nRuns),AAG)
   x = checkRelaunch(name)   
   if x==0:
      print("Run not found in relaunch!!!")
      relaunch(folder,name)
   elif x==1:
      print("file in relaunch list")
      remove(folder)
   else:
      print("Unable to check if file in relaunch")

def remove(folder):
   os.system("rm -r %s"%folder)
   print("deleted !")

def relaunch(folder,log):
   log = log.split(" AAG = ")
   if log[1]=="False":
      relaunchFile = "FailledRun.txt"
   elif log[1]=="True":
      relaunchFile = "FailledRun_Aag.txt"
   else:
      print("ERROR, unable to get run type")
      return (-1)
   os.system("echo %s >> %s"%(log[0],relaunchFile))
   print("Added to relaunch list (%s)."%relaunchFile)
   remove(folder)

def getCollection(log):
   for x in log:
      if "ALCARECOSiStripCalMinBias" in x:
         return("Std")
      if "ALCARECOSiStripCalMinBiasAfterAbortGap" in x:
         return("Aag")
   return ("None")



NoError = []
Error = []
for folder in os.listdir("."):
   if "core." in folder[:5]:
      print("removing %s"%folder)
      os.system("rm %s"%folder)
   if not folder[:3]=="LSF" or not os.path.isdir(folder):
      continue
   if not "STDOUT" in os.listdir(folder):
      print("Error, no STDOUT file in folder %s !"%folder)
      continue 
   log = open(folder+"/STDOUT","r").read()
   
   if "stageout" in log[-2000:]:
      NoError.append(folder)
   else:
      Error.append(folder)

if len(NoError)+len(Error)==0:
   print("Nothing to do...")
ToKeep = []

for f in NoError:
   log = open(f+"/STDOUT","r").read()
   if "WARNING WARNING WARNING STAGE OUT FAILED BUT NOT RELAUNCHED" in log:
      Error.append(f)
   elif "The file size is" in log[-2000:]:
      print("Removing good run %s - %s"%(f,getName(log.split("\n"))))
      remove(f)
   else:
      print("Something fishy in %s (marked as good)"%f)
      ToKeep.append(f)

for f in Error:
   toRemove=False
   log= open(f+"/STDOUT","r").read()
   log=log.split("\n")
   eMessage=""
   if len(log)<80:
      print("Short in %s (%s)"%(f,len(log)))
      relaunchShort(f)
      logLen=len(log)
   else:
      logLen=80
   for i in range(logLen):
      if "Disk quota exceeded" in log[-i]:
         eMessage = "Disk quota exceeded"
         toRemove=True
         break
      elif "cms.untracked.vstring('ProductNotFound')" in log[-i]:
         eMessage = "Product not found"
         toRemove = True
         break
      elif "Can not interpret the query (while creating DASQuery)" in log[-i]:
         eMessage = "DAS query not understood"
         toRemove=False
         break
      elif "client timeout after" in log[-i]:
         eMessage = "DAS timeout"
         toRemove=False
      elif "No such file or directory" in log[-i]:
         eMessage = "Can't open input file"
         toRemove=True
         break
      elif "Network dropped connection on reset" in log[-i]:
         eMessage = "Network dropped connection on reset"
         toRemove=True
         break
      elif "Job Failed with ExitCode" in log[-i]:
         eMessage= log[-i]
         toRemove=True
         break
      elif "WARNING WARNING WARNING STAGE OUT FAILED BUT NOT RELAUNCHED" in log[-i]:
         eMessage= "Bad stageout status."
         toRemove=True
         break
   if not eMessage == "" and toRemove:
      name = getName(log)
      if "Unable" in name:
         print("%s - %s"%(f,name))
         continue
      print("Removing bad run %s - %s (%s)"%(f,eMessage,name))
      code = checkRelaunch(name)
      if code==-1:
         print("ABORT, unable to get relaunch status")
      elif code==0:
         relaunch(f,name)
      else:
         print("Found in relaunch list... deleting...")
         remove(f)
   else:
      print("Something fishy in %s"%f)
      ToKeep.append(f)

if len(ToKeep)>0:
   print("Strange jobs :")
   for i in ToKeep: print(i)
