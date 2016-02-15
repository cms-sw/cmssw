#!/usr/bin/env python

import string
import os
import sys
import subprocess

dataDir = ''


def GetNewCfgName(runNumber):
  return 'hcalSourceDataMon.'+str(runNumber)+'_cfg.py'

def GetPlotCfgName(runNumber):
  return 'hcalSourceDataMonPlots.'+str(runNumber)+'.py'

def GetNewNTupleFileName(runNumber):
  return 'hcalSourceDataMon.'+str(runNumber)+'.root'

def GetPlotFileName(runNumber):
  return 'hcalSourceDataMonPlots.'+str(runNumber)+'.root'

def GetFilenames(runNumber):
  global dataDir
  fileNames = ''
  oldDir = os.getcwd()
  os.chdir(dataDir)
  dirFileList = os.listdir(".")
  dirFileList.sort()
  dirFileList.sort(key=len) # put shortest file name (without .1, .2, etc., at beginning)
  for files in dirFileList:
    if files.endswith(".root") and str(runNumber) in files:
      fileNames+=("      'file:"+dataDir+"/"+files+"',\n")
  os.chdir(oldDir)
  print 'using files:'
  print fileNames
  return fileNames


def CreateNTupleConfigFile(runNumber):
  Path_Cfg = 'hcalsourcedatamonitor_template_cfg.py'
  newCfgName = GetNewCfgName(runNumber)
  config_file=open(Path_Cfg,'r')
  config_txt = config_file.read()
  config_file.close()
  # Replacements
  config_txt = config_txt.replace("XXX_TFILENAME_XXX", GetNewNTupleFileName(runNumber))
  config_txt = config_txt.replace("XXX_FILENAMES_XXX", GetFilenames(runNumber))
  # write
  config_file=open(str(runNumber)+'/'+newCfgName,'w')
  config_file.write(config_txt)
  config_file.close()


def CreatePlotsConfigFile(runNumber):
  Path_Cfg = 'hcalsourcedatamonplots_template.py'
  newCfgName = GetPlotCfgName(runNumber)
  config_file=open(Path_Cfg,'r')
  config_txt = config_file.read()
  config_file.close()
  # Replacements
  config_txt = config_txt.replace("XXX_NTUPLEFILE_XXX", GetNewNTupleFileName(runNumber))
  config_txt = config_txt.replace("XXX_PLOTFILE_XXX", GetPlotFileName(runNumber))
  # write
  config_file=open(str(runNumber)+'/'+newCfgName,'w')
  config_file.write(config_txt)
  config_file.close()


def CreateCfgs(runNumber):
  if not os.path.isdir(str(runNumber)):
    os.mkdir(str(runNumber))
  CreateNTupleConfigFile(runNumber)
  CreatePlotsConfigFile(runNumber)


# RUN
runsList = [
    217090
]

dataDir = "/data2/scooper/Sourcing/December_HEM09_10_11"
#dataDir = "/data2/scooper/Sourcing/P5_Tests_JustBefore_December2013HEM"
#dataDir = "/data2/scooper/Sourcing/P5_Tests_JustBefore_October2013HFM"
#dataDir = "/data2/scooper/Sourcing/AfterOctoberHFM_Tests"
#dataDir = "/data2/scooper/Sourcing/October_HFM_Q1_Q4"
#dataDir = "/mnt/bigspool/usc"

for run in runsList:
  print 'Run number:',run
  CreateCfgs(run)
  print 'cmsRun',GetNewCfgName(run),'from:',os.getcwd()+'/'+str(run)
  proc = subprocess.Popen(['cmsRun',GetNewCfgName(run)],cwd=os.getcwd()+'/'+str(run),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  out,err = proc.communicate()
  print out
  print err
  proc = subprocess.Popen(['hcalSourceDataMonPlots',GetPlotCfgName(run)],cwd=os.getcwd()+'/'+str(run),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  out,err = proc.communicate()
  print out
  print err
  





