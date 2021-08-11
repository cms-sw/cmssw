#!/usr/bin/env python3
from __future__ import print_function
import os, time, sys, glob, re, smtplib, socket
from email.MIMEText import MIMEText
from traceback import print_exc, format_exc
from datetime import datetime
from subprocess import Popen,PIPE
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

EMAIL = sys.argv[1]
TFILEDONEDIR = sys.argv[2]
COLLECTDIR = sys.argv[3]
ORIGINALDONEDIR =sys.argv[4]

#Constans
PRODUCER_DU_TOP= 90.0  #0% a 100%
PRODUCER_DU_BOT= 50.0  #0% a 100%
WAITTIME = 3600 * 4
EMAILINTERVAL = 15 * 60 # Time between sent emails 
SENDMAIL = "/usr/sbin/sendmail" # sendmail location
HOSTNAME = socket.gethostname().lower()
EXEDIR = os.path.dirname(__file__)
STOP_FILE = "%s/.stop" % EXEDIR

# Control variables
lastEmailSent = 0

# --------------------------------------------------------------------
def logme(msg, *args):
  procid = "[%s/%d]" % (__file__.rsplit("/", 1)[-1], os.getpid())
  print(datetime.now(), procid, msg % args)
  
def getDiskUsage(path):
  fsStats=os.statvfs(path)
  size=fsStats.f_bsize*fsStats.f_blocks
  available=fsStats.f_bavail*fsStats.f_bsize
  used=size-available
  usedPer=float(used)/size
  return (size,available,used,usedPer)
  
def getDirSize(path): 
  import stat
  size=os.stat(path).st_blksize
  for directory,subdirs,files in os.walk(path):
    dStats=os.lstat(directory)
    size+=(dStats[stat.ST_NLINK]-1)*dStats[stat.ST_SIZE]
    for f in files:  
      fStats=os.lstat("%s/%s" % (directory,f))
      fSize=fStats[stat.ST_SIZE]
      size+=fSize
      
  return size
  
def sendmail(body="Hello from producerFileCleanner",subject= "Hello!"):
  scall = Popen("%s -t" % SENDMAIL, shell=True, stdin=PIPE)
  scall.stdin.write("To: %s\n" % EMAIL)
  scall.stdin.write("Subject: producerFileCleaner problem on server %s\n" %
                     HOSTNAME)
  scall.stdin.write("\n") # blank line separating headers from body
  scall.stdin.write("%s\n" % body)
  scall.stdin.close()
  rc = scall.wait()
  if rc != 0:
     logme("ERROR: Sendmail exit with status %s", rc)
  
# --------------------------------------------------------------------    
while True:
  #Check if you need to stop.
  if os.path.exists(STOP_FILE):
    logme("INFO: Stop file found, quitting")
    sys.exit(0)

  try:
    try:
      doneSize=getDirSize(TFILEDONEDIR)
      diskSize,userAvailable,diskUsed,diskPUsage=getDiskUsage(TFILEDONEDIR)
      
    except:
      doneSize=0
      diskSize,userAvailable,diskUsed,diskPUsage=getDiskUsage("/home")
      
    diskPUsage*=100
    if diskPUsage < PRODUCER_DU_TOP:
      time.sleep(WAITTIME)
      continue
      
    quota=long(diskSize*PRODUCER_DU_BOT/100)
    delQuota=diskUsed-quota
    if delQuota > doneSize:
      now = time.time()
      if now - EMAILINTERVAL > lastEmailSent:
        msg="ERROR: Something is filling up the disks, %s does not" \
          " have enough files to get to the Bottom Boundary of" \
          " %.2f%%" % (TFILEDONEDIR,PRODUCER_DU_BOT)
        sendmail(msg)
        lastEmailSent = now
        
      logme("ERROR: Something is filling up the disks, %s does not" \
          " have enough files to get to the Bottom Boundary of" \
          " %.2f%%", TFILEDONEDIR, PRODUCER_DU_BOT)
      
    aDelQuota=0
    FILE_LIST=[]
    for directory,subdirs,files in os.walk(TFILEDONEDIR):
      subdirs.sort()
      for f in sorted(files,key=lambda a: a[a.rfind("_R",1)+2:a.rfind("_R",1)+11]):
        fMatch=re.match(r"(DQM|Playback|Playback_full)_V[0-9]{4}_([0-9a-zA-Z]+)_R([0-9]{9})(_T[0-9]{8}|)\.root",f)
        if fMatch:
          subSystem=fMatch.group(2)
          run=fMatch.group(3)
          destDir="%s/%sxxxx/%sxx/DQM_V0001_%s_R%s.root" % (ORIGINALDONEDIR,run[0:5],run[0:7],subSystem,run)
          fullFName="%s/%s" % (directory,f)
          if os.stat(fullFName).st_size+aDelQuota > delQuota:
            break
          
          FILE_LIST.append(fullFName)
          aDelQuota+=os.stat(fullFName).st_size
          if not os.path.exists(destDir):
            logme("WARNING: No subsystem file in repository %s for"
                  " file %s, deleting any way" % 
                  (ORIGINALDONEDIR, fullFName))
            
    if len(FILE_LIST):
      logme("INFO: Found %d files to be deleted", len(FILE_LIST))
      
    #Cleanning ouput directory
    for directory,subdirs,files in os.walk(COLLECTDIR):
      #no subdiretories allowed in COLLECTDIR the  directory
      if subdirs:
        logme("ERROR: Output directory %s, must not contain"
              " subdirectories, cleanning", COLLECTDIR)
        
      for sd in subdirs:
        fullSdName="%s/%s" % (directory,sd)
        for sdRoot,sdDirs,sdFiles in os.walk(fullSdName,topdown=False):
          for f in sdFiles:
            try:
              os.remove(f)
              logme("INFO: File %s has been removed", f)
            except Exception as e:
              logme("ERROR: Problem deleting file: [Errno %d] %s, '%s'",
                      e.errno, e.strerror, e.filename)
              
          try:
            os.removedir(sdRoot)
            logme("INFO: File %s has been removed" , sdRoot)
          except Exception as e:
            logme("ERROR: Problem deleting directory: [Errno %d] %s, '%s'",
                      e.errno, e.strerror, e.filename)
                      
      for f in files:
        if re.match(r"(DQM|Playback|Playback_full)_V[0-9]{4}_([a-zA-Z]+)_R([0-9]{9})_T[0-9]{8}\.root", f):
          continue
          
        if re.match(r".*\.tmp",f):
          continue
          
        fullFName="%s/%s" % (directory, f)
        FILE_LIST.append(fullFName)
        
      #cleaning tmp files:
      TMP_LIST=glob.glob("%s/*.tmp" % COLLECTDIR)
      TMP_LIST.sort(reverse=True,key=lambda x: os.stat(x).st_mtime)
      len(TMP_LIST) > 0 and TMP_LIST.pop(0)
      FILE_LIST.extend(TMP_LIST)
      
    #remove files
    DIR_LIST=[]
    for f in FILE_LIST:
      try:
        os.remove(f)
        logme("INFO: File %s has been removed", f)
      except Exception as e:
        logme("ERROR: Problem deleting file: [Errno %d] %s, '%s'",
                e.errno, e.strerror, e.filename)
      if os.path.dirname(f) not in DIR_LIST and COLLECTDIR not in os.path.dirname(f):
        DIR_LIST.append(os.path.dirname(f))
        
    #remove emprty directories
    for d in DIR_LIST:
      try:
        os.removedirs(d)
        logme("INFO: Directory %s has been removed", d)
      except Exception as e:
        logme("ERROR: Directory delition failed: [Errno %d] %s, '%s'",
                e.errno, e.strerror, e.filename)

  except KeyboardInterrupt as e:
    sys.exit(0)

  except Exception as e:
    logme('ERROR: %s', e)
    sendmail ('ERROR: %s\n%s' % (e, format_exc()))
    now = time.time()
    if now - EMAILINTERVAL > lastEmailSent:
      sendmail ('ERROR: %s\n%s' % (e, format_exc()))
      lastEmailSent = now
    
    print_exc() 
  
  time.sleep(WAITTIME)               

         
      
    
