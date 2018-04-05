#!/usr/bin/env python
import os, commands

class configuration:
   datasetPat  = '/StreamExpress/Run2017*-SiStripCalMinBias__AAG__-Express-v*/ALCARECO'
   CMSSWDIR    = 'TO_FILL_IN'
   RUNDIR      = CMSSWDIR+'CalibTracker/SiStripCommon/test/MakeCalibrationTrees/'
   CASTORDIR   = '/store/group/dpg_tracker_strip/comm_tracker/Strip/Calibration/calibrationtree/GR17__AAG__'
   nFilesPerJob= 25
   collection  = "ALCARECOSiStripCalMinBias__AAG__"
   globalTag   = "92X_dataRun2_Express_v2"
   initEnv     = ""
   eosLs       = "eos ls "
   def  __init__(self,AAG=False,debug=False):
      self.relaunchList= []
      self.firstRun    = -1
      self.lastRun     = 999999
      self.launchedRuns = []
      self.AAG          = AAG
      self.datasetPat   = self.datasetPat.replace("__AAG__","AAG" if self.AAG else "")
      self.CASTORDIR    = self.CASTORDIR.replace ("__AAG__","_Aag" if self.AAG else "")
      self.collection   = self.collection.replace("__AAG__","AAG" if self.AAG else "")
      self.initEnv+='cd ' + self.CMSSWDIR + '; '
      self.initEnv+='export CMS_PATH=/cvmfs/cms.cern.ch; '
      self.initEnv+='source /afs/cern.ch/cms/cmsset_default.sh' + ';'
      self.initEnv+='eval `scramv1 runtime -sh`' + ';'
      self.initEnv+='cd -;'
      self.submit = not debug
      self.integrity = False
      self.setupEnviron()
      print "Integrity = %s"%self.checkIntegrity()

   def checkIntegrity(self):
      goodConfig=True

      #Check dataset :
      d = self.datasetPat.split("/")
      if not len(d) == 4:
         print "Bad dataset. Expecting 4 '/'"
         goodConfig=False
      if not d[0]=='':
         print "Bad dataset. Expecting nothing before first '/'"
         goodConfig=False
      if not len(d[1])>0 or not len(d[2]) > 0 or not len(d[3]) > 0:
         print "Bad dataset. Expecting text between '/'"
         goodConfig=False
      if os.path.isdir(self.datasetPat):
         print "Bad dataset. Can't be an existing directory"
         goodConfig=False
      #Check all paths exist
      if not os.path.isdir(self.CMSSWDIR):
         print "CMSSW dir does not exist."
         goodConfig = False
      if not os.path.isdir(self.RUNDIR):
         print "RUN dir does not exist."
         goodConfig = False

      #Check castor path exists FIXME
      cmd = self.eosLs.replace("-lrth","")+self.CASTORDIR
      cmd = cmd[:-2]+"*"
      (status,output) = commands.getstatusoutput(cmd)
      if status or not self.CASTORDIR.split("/")[-1] in output:
         print cmd
         print output
         print "CASTOR dir does not exist."
         goodConfig = False
      self.integrity = goodConfig
      return goodConfig   

   def setupEnviron(self):
      os.environ['PATH'] = os.getenv('PATH')+':/afs/cern.ch/cms/sw/common/'
      os.environ['CMS_PATH']='/afs/cern.ch/cms'
      os.environ['FRONTIER_PROXY'] = 'http://cmst0frontier.cern.ch:3128'
      os.environ['SCRAM_ARCH']='slc6_amd64_gcc530'
   def __str__(self):
      description = "Configuration :\n"
      description+= "First run  = %s\n"   %self.firstRun
      description+= "After Abort= %s\n"   %self.AAG
      description+= "dataset    = %s\n"   %self.datasetPat
      description+= "CMSSW      = %s\n"   %self.CMSSWDIR
      description+= "RUNDIR     = %s\n"   %self.RUNDIR
      description+= "CASTOR     = %s\n"   %self.CASTORDIR
      description+= "nFiles     = %s\n"   %self.nFilesPerJob
      description+= "collection = %s\n"   %self.collection
      description+= "initEnv    = %s\n"   %self.initEnv
      description+= "submit     = %s\n"   %self.submit
      return description

if __name__ == "__main__":
   c = configuration(True)
   print c


