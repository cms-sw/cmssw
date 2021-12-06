#! /usr/bin/env python3

from __future__ import print_function
from builtins import range
import os,time,sys,zipfile,re,shutil,stat
from fcntl import lockf, LOCK_EX, LOCK_UN
from hashlib import md5
from glob import glob
from datetime import datetime

COLLECTING_DIR = sys.argv[1] #Directory where to look for root files
T_FILE_DONE_DIR = sys.argv[2] #Directory where to place processed root files
DROPBOX = sys.argv[3] #Directory where the collected files are sent.

EXEDIR = os.path.dirname(__file__) 
COLLECTOR_WAIT_TIME = 10 # time  between collector cilces
WAIT_TIME_FILE_PT = 60 * 15 # time to wait to pick up lost files
TMP_DROPBOX = os.path.join(DROPBOX,".uploading")
KEEP = 2 # number of _d files to keep
RETRIES = 3 # number of retries to sen a file
STOP_FILE = "%s/.stop" % EXEDIR
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
os.environ["WorkDir"] = EXEDIR

def logme(msg, *args):
  procid = "[%s/%d]" % (__file__.rsplit("/", 1)[-1], os.getpid())
  print(datetime.now(), procid, msg % args)
  
def filecheck(rootfile):
  cmd = EXEDIR + '/filechk.sh ' + rootfile
  a = os.popen(cmd).read().split()
  tag=a.pop()
  if tag == '(int)(-1)':
    logme("ERROR: File %s corrupted (isZombi)", rootfile)
    return False
  elif tag == '(int)0':
    logme("ERROR: File %s is incomplete", rootfile)
    return False
  elif tag == '(int)1':
    return True
  else:
    return False

def isFileOpen(fName):
  fName = os.path.realpath(fName)
  pids=os.listdir('/proc')
  for pid in sorted(pids):
    try:  
      if not pid.isdigit():
        continue
    
      if os.stat(os.path.join('/proc',pid)).st_uid != os.getuid():
        continue
      
      uid = os.stat(os.path.join('/proc',pid)).st_uid
      fd_dir=os.path.join('/proc', pid, 'fd')
      if os.stat(fd_dir).st_uid != os.getuid():
        continue
        
      for f in os.listdir(fd_dir):
        fdName = os.path.join(fd_dir, f)
        if os.path.islink(fdName) :       
            link=os.readlink(fdName)
            if link == fName:
              return True
    except:
      continue
          
  return False
     
def convert(infile, ofile):
  cmd = EXEDIR + '/convert.sh ' + infile + ' ' +ofile
  os.system(cmd)
  
def uploadFile(fName, subsystem, run):
  hname = os.getenv("HOSTNAME")
  seed=hname.replace("-","t")[-6:]
  finalTMPfile="%s/DQM_V0001_%s_R%s.root.%s" % (TMP_DROPBOX,subsystem,run,seed)        
  if os.path.exists(finalTMPfile):
    os.remove(finalTMPfile)
            
  md5Digest=md5(file(fName).read())
  originStr="md5:%s %d %s" % (md5Digest.hexdigest(),os.stat(fName).st_size,fName)
  originTMPFile="%s.origin" % finalTMPfile
  originFile=open(originTMPFile,"w")
  originFile.write(originStr)
  originFile.close() 
  shutil.copy(fName,finalTMPfile)
  if not os.path.exists(finalTMPfile) or not os.stat(finalTMPfile).st_size == os.stat(fName).st_size:
    return False
  
  version=1
  lFile=open("%s/lock" % TMP_DROPBOX ,"a")
  lockf(lFile,LOCK_EX)
  for vdir,vsubdir,vfiles in os.walk(DROPBOX):
    if 'DQM_V0001_%s_R%s.root' % (subsystem,run) not in vfiles:
      continue
    
    version += 1

  if not os.path.exists("%s/%04d" % (DROPBOX,version)):
    os.makedirs("%s/%04d" % (DROPBOX,version))
    os.chmod("%s/%04d" % (DROPBOX,version),2775)
                  
  finalfile="%s/%04d/DQM_V0001_%s_R%s.root" %   (DROPBOX,version,subsystem,run)        
  originFileName="%s.origin" % finalfile     
  try:
    os.rename(finalTMPfile,finalfile)
    os.rename(originTMPFile,originFileName)
    os.chmod(finalfile,stat.S_IREAD|stat.S_IRGRP|stat.S_IROTH| stat.S_IWRITE|stat.S_IWGRP|stat.S_IWOTH)
    os.chmod(originFileName,stat.S_IREAD|stat.S_IRGRP|stat.S_IROTH| stat.S_IWRITE|stat.S_IWGRP|stat.S_IWOTH)  
  except:
    lockf(lFile,LOCK_UN)
    lFile.close()
    logme("ERROR: File %s upload failed to the DROPBOX" % fName)
    return False
    
  logme("INFO: File %s has been successfully sent to the DROPBOX" % fName)
  lockf(lFile,LOCK_UN)
  lFile.close()
  return True
  
def processSiStrip(fName,finalTfile):
  dqmfile = fName
  if "Playback" in fName and "SiStrip" == NEW[rFile]["subSystem"]:
    dqmfile = fName.replace('Playback','DQM')
    convert(fName,dqmfile)
    if not os.path.exists(dqmfile):
      logme("ERROR: Problem converting %s skiping" % Tfile)
      shutil.move(fName,finalTfile+"_d")
      return (dqmfile,False)
      
    os.rename(fName,finalTfile.replace('Playback','Playback_full'))
  
  return (dqmfile,True)
  
####### ENDLESS LOOP WITH SLEEP
NEW = {}
LAST_SEEN_RUN = "0"
LAST_FILE_UPLOADED = time.time()
if not os.path.exists(TMP_DROPBOX):
  os.makedirs(TMP_DROPBOX)

while True:
  #Check if you need to stop.
  if os.path.exists(STOP_FILE):
    logme("INFO: Stop file found, quitting")
    sys.exit(0)

  #clean up tagfiele_runend files, this should be removed as it use is deprecated
  TAGS=sorted(glob('%s/tagfile_runend_*' % COLLECTING_DIR ),reverse=True)
  for tag in TAGS:
    os.remove(tag)
    
  for dir, subdirs, files in os.walk(COLLECTING_DIR):
    for f in files:
      fMatch=re.match('^(DQM|Playback)_V[0-9]{4}_(?P<subsys>.*)_R(?P<runnr>[0-9]{9})\.root$',f)
      if fMatch:
        runnr = fMatch.group("runnr")
        subsystem=fMatch.group("subsys")
        f = "%s/%s" % (dir, f)
        NEW.setdefault(f, {"runNumber":runnr, 
                           "subSystem":subsystem, 
                           "Processed":False, 
                           "TFiles":[]})
        if int(runnr) > int(LAST_SEEN_RUN):
          LAST_SEEN_RUN = runnr
  
  for rFile in NEW.keys():
    if len(NEW[rFile]["TFiles"]):
      continue
      
    # Add respective T files just in case the final root file is damage
    for dir, subdirs, files in os.walk(COLLECTING_DIR):
      for f in files:
        runnr = NEW[rFile]["runNumber"]
        subsystem=NEW[rFile]["subSystem"]
        fMatch=re.match('^(DQM|Playback)_V[0-9]{4}_%s_R%s_T[0-9]{8}.root$' % (
                    subsystem, runnr),f)
        if fMatch:
          f = "%s/%s" % (dir, f)
          NEW[rFile]["TFiles"].append(f)

      NEW[rFile]["TFiles"].sort(reverse=True)   
  
  #Process files
  for rFile in NEW.keys():
    if isFileOpen(rFile):
      logme("INFO: File %s is open", rFile)
      continue
    
    transferred = False
    run = NEW[rFile]["runNumber"]
    subsystem = NEW[rFile]["subSystem"]
    finalTdir="%s/%s/%s" % (T_FILE_DONE_DIR,run[0:3],run[3:6])
    if not os.path.exists(finalTdir):
      os.makedirs(finalTdir)
    
    if not filecheck(rFile):
      os.rename(rFile,"%s/%s_d" % (finalTdir, os.path.basename(rFile)))
      for Tfile in NEW[rFile]["TFiles"]:
        finalTfile="%s/%s" % (finalTdir,os.path.basename(Tfile))
        if transferred:
          break
        
        if not filecheck(Tfile):
          if os.path.exists(Tfile):
            shutil.move(Tfile,finalTfile+"_d")
          continue
            
        fToUpload, converted = processSiStrip(Tfile, finalTfile)
        if not converted:
          continue

        for i in range(RETRIES):
          if uploadFile(fToUpload, subsystem, run):
            NEW[rFile]["Processed"] = transferred = True
            LAST_FILE_UPLOADED = time.time()
            os.rename(fToUpload, "%s/%s" % (finalTdir, os.path.basename(fToUpload)))
            break
        
      NEW[rFile]['Processed'] = True
      continue
    
    finalTfile="%s/%s" % (finalTdir,os.path.basename(rFile))    
    fToUpload, converted = processSiStrip(rFile, finalTfile)
    if not converted:
      continue
    
    for i in range(RETRIES):
      if uploadFile(fToUpload, subsystem, run):
        NEW[rFile]["Processed"] = transferred = True
        LAST_FILE_UPLOADED = time.time()
        os.rename(fToUpload, "%s/%s" % (finalTdir, os.path.basename(fToUpload)))
        break
  
  #Clean up COLLECTING_DIR
  for rFile in NEW.keys():
    if not  NEW[rFile]["Processed"]:
      continue

    run = NEW[rFile]["runNumber"]
    subsystem = NEW[rFile]["subSystem"]
    finalTdir="%s/%s/%s" % (T_FILE_DONE_DIR,run[0:3],run[3:6])
    for Tfile in NEW[rFile]["TFiles"]:
      if os.path.exists(Tfile):
        finalTfile="%s/%s_d" % (finalTdir,os.path.basename(Tfile)) 
        os.rename(Tfile,finalTfile)
        
    #Enforce KEEPS
    fList = sorted(glob("%s/*_%s_R%s*_d" % (finalTdir,subsystem, run)),cmp=lambda x,y: "_T" not in x and 1 or ("_T" in y and ( -1 * cmp(x,y))))
    for f in fList[::-1]:
      if len(fList) > KEEP:
        fList.remove(f)
        os.remove(f)
  
  #Determine if the run has been fully processed.
  for rFile in NEW.keys():
    if NEW[rFile]['Processed']:
      del NEW[rFile]
      
  #Find and process orphan _T files.
  if LAST_FILE_UPLOADED < time.time() - WAIT_TIME_FILE_PT:
    for dir, subdirs, files in os.walk(COLLECTING_DIR):
      for f in files:
        fMatch=re.match('^(DQM|Playback)_V[0-9]{4}_(?P<subsys>.*)_R(?P<runnr>[0-9]{9})_T[0-9]{8}\.root$',f)
        if not fMatch:
          continue
          
        runnr = fMatch.group("runnr")
        subsystem=fMatch.group("subsys")
        if runnr > LAST_SEEN_RUN:
          continue
          
        tmpFName = "%s/%s.root" % (dir,f.rsplit("_",1)[0])
        if os.path.exists(tmpFName):
          continue
          
        finalTdir = "%s/%s/%s" % (T_FILE_DONE_DIR,runnr[0:3],runnr[3:6])
        fList = sorted(glob("%s/*_%s_R%s*" % (finalTdir,subsystem, runnr)),
                    cmp=lambda x,y: cmp(os.stat(x).st_mtime,os.stat(y).st_mtime))
        fName = "%s/%s" % (dir,f)
        if len(fList) and os.stat(fList[-1]).st_mtime > os.stat(fName).st_mtime:
          os.remove(fName)
          continue
        
        logme("INFO: Creating dummy file %s to pick up Orphan _T files", tmpFName)
        tmpF = open(tmpFName,"w+")
        tmpF.close()
        del tmpF      
        
  time.sleep(COLLECTOR_WAIT_TIME)
