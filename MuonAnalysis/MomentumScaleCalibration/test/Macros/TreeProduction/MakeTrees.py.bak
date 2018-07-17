# -*- coding: utf-8 -*-

# Multithreaded python script that can be used to run parallel cmssw jobs to produce the MuScleFit trees
# Change the numberOfThreads value (set to three by default) and the castor dir and run it with python.
# It will run a job per file spawning no more than numberOfThreads simultaneus threads.
# At the end it will produce a MergeTrees.cc macro that can be used to merge all the trees together.
# It Uses a MuScleFit_template_cfg file to create the MuScleFit cfg with the correct file name.

# Important parameters
# --------------------

# Number of simultaneous threads 
TotalNumberOfThreads = 3
# Directory on castor with the input files
CastorFilesDir = "/castor/cern.ch/user/d/demattia/MuScleFit/Summer10/JPsi/ModifiedMaterialScenario/OniaPAT"
# Directory where to do eval scram
CMSSWDir = "/afs/cern.ch/user/d/demattia/scratch0/TreeProducerAndMerger/CMSSW_3_8_0/src"

# --------------------

import os
import commands

# Lock and take the new index
# increase it
# Unlock it
# Now execute cmsRun
      
# Example from here: http://www.ibm.com/developerworks/aix/library/au-threadingpython/index.html

import Queue
import threading
import urllib2
import time
          
queue = Queue.Queue()
          
class ThreadUrl(threading.Thread):
  """The thread that will launch the cmsRun"""
  def __init__(self, queue):
    threading.Thread.__init__(self)
    self.queue = queue

  def run(self):
    while True:
      #grab file name from queue
      data = self.queue.get()
      filesDir = data[2]
      inputFileName = data[0].split()[-1]
      num = str(data[1])
      outputFileName = data[3]
      print "input file name = ", inputFileName, " output file name = ", outputFileName
      templateCfg = open("MuScleFit_template_cfg.py").read()
      templateCfg = templateCfg.replace("INPUTFILENAME", "rfio:"+filesDir+inputFileName).replace("OUTPUTTREENAME", outputFileName)
      cfgName = "MuScleFitTree_cfg_"+num+".py"
      cfg = open(cfgName, "w")
      cfg.write(templateCfg)
      cfg.close()

      # Run the cmssw job
      print "cd "+CMSSWDir+"; eval `scramv1 r -sh`; cd -; cmsRun "+cfgName
      os.system("cd "+CMSSWDir+"; eval `scramv1 r -sh`; cd -; cmsRun "+cfgName)
      os.system("mv "+cfgName+" processedCfgs")

      #signals to queue job is done
      self.queue.task_done()

def main(numberOfThreads):

  # Take the files
  filesDir = CastorFilesDir
  if not filesDir.endswith("/"):
    filesDir = filesDir+"/"
  os.system("rfdir "+filesDir+" > list.txt")
  f = open("list.txt")
  os.system("mkdir -p processedCfgs")

  # Prepare the macro to merge the trees
  macro = open("MergeTrees.cc", "w")
  macro.write("#include <iostream>\n")
  macro.write("#include <TFile.h>\n")
  macro.write("#include <TChain.h>\n")
  macro.write("void MergeTrees()\n")
  macro.write("{\n")
  macro.write("  TChain * chain = new TChain(\"T\");\n")


  #spawn a pool of threads, and pass them queue instance 
  for i in range(numberOfThreads):
    t = ThreadUrl(queue)
    t.setDaemon(True)
    t.start()
              
  #populate queue with data   
  num = 0
  for line in f:
    outputFileName = "tree_"+str(num)+".root"
    macro.write("  chain->Add(\""+outputFileName+"\");\n")
    lineAndNum = [line, num, filesDir, outputFileName]
    queue.put(lineAndNum)
    num += 1


  # All threads ready, close the macro
  macro.write("  chain->Merge(\"fullTree.root\");\n")
  macro.write("}\n")
  macro.close()

  #wait on the queue until everything has been processed     
  queue.join()

# Run the jobs withTotalNumberOfThreads threads
start = time.time()
main(TotalNumberOfThreads)
print "Elapsed Time: %s" % (time.time() - start)
