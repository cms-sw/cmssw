#!/usr/bin/env python3

from __future__ import print_function
import os, time, sys, shutil, glob, smtplib, re, subprocess
import getopt as gop
import zipfile as zp
from datetime import datetime
from email.MIMEText import MIMEText

SOURCEDIR = "/dqmdata/dqm/merged" #Directory where the original files are located.
INJECTIONDIR = "/dqmdata/dqm/Tier0Shipping/inject"   #Directory where files get placed once they have been sent.
TRASHCAN = "/dqmdata/dqm/trashcan"
HOSTNAME = "srv-C2D05-19"
CONFIGFILE = "/nfshome0/dqm/.transfer/myconfig.txt"
INJECTIONSCRIPT = "/nfshome0/tier0/scripts/injectFileIntoTransferSystem.pl"

EMPTY_TRASHCAN = False
WAIT_TIME = 600
################################################################################
def getFileLists():
  newfiles={}
  trashfiles=[]
  filelist=[]
  for dir1, subdirs, files in os.walk(SOURCEDIR):
    for f in files:
      try:
        version=int(f.split("_V")[-1][:4])
        vlfn=f.split("_V")[0]+f.split("_V")[1][4:]
      except:
        version=1
        vlfn=f
      if vlfn in newfiles.keys() and version >= newfiles[vlfn][0]:
        newfiles[vlfn]=(version,"%s/%s" % (dir1,f))
      elif vlfn not in newfiles.keys():
        newfiles[vlfn]=(version,"%s/%s" % (dir1,f))
      else:
        trashfiles.append("%s/%s" % (dir1,f))
      
  return (newfiles,trashfiles)
#=====================================================================================
def injectFile(f,renotify=False):
  fname=f.rsplit("/",1)[-1]
  dname=f.rsplit("/",1)[0]
  run=f.split("_R")[-1][:9]
  iname="%s/%s" % (INJECTIONDIR,fname)
  shutil.move(f,iname)
  parameters=["--filename %s" % fname,
              "--type dqm",
              "--path %s" % INJECTIONDIR,
              "--destination dqm",
              "--hostname %s" % HOSTNAME,
              "--config %s" % CONFIGFILE,
              "--runnumber %s" % run,
              "--lumisection 95",
              "--numevents 834474816",
              "--appname dqmArchive",
              "--appversion dqmArchive_1_0"]
  cmd="%s %s" % (INJECTIONSCRIPT," ".join(parameters))
  result = subprocess.getstatusoutput(cmd)
  if result[0] >= 1:
    output = result[1]
    print("Error injecting file %s to transfer system checking if it exists" % f)
    chkparameters=["--check","--filename %s" % fname,"--config %s" % CONFIGFILE]
    cmd="%s %s" % (INJECTIONSCRIPT," ".join(chkparameters))
    result = subprocess.getstatusoutput(cmd)
    if result[0]==1:
      if "File not found in database" in result[1]:
        print("Error: file %s not found in transfer database, check configuration" % f)
        return 0
      else:
        print("Warning: file %s already exists in transfer database" % f)
        return 2
    else:
      print("Error: problem checking database entry for file %s\n Error:%s" % (f,result[1]))
      return 0
  else:
    print("File %s injected successfully" % f)
  return 1
#=====================================================================================
def transferFiles():
  while True:
    #look for NEW files in SOURCEDIR and files that need to be cleared.
    newfiles,trashfiles=getFileLists() 
    
    #Making sure destination directories exist
    if not os.path.exists(TRASHCAN):
      os.makedirs(TRASHCAN)
    if not os.path.exists(INJECTIONDIR):
      os.makedirs(INJECTIONDIR)
      
    #Dealing with trash can  
    for tf in trashfiles:
      if EMPTY_TRASHCAN:
        os.remove(tf)
      else:
        tfname="%s/%s" % (TRASHCAN,tf.rsplit("/",1)[-1])
        shutil.move(tf,tfname)
        
    #Down to bussines
    for ver,f in newfiles.values():
      ifr=injectFile(f)
    time.sleep(WAIT_TIMEt	)
#=====================================================================================
if __name__ == "__main__": 
  transferFiles()
