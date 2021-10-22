#! /usr/bin/env python3
from __future__ import print_function
from builtins import range
import os, time, sys, glob, re, shutil, stat, smtplib, socket
from email.MIMEText import MIMEText
from fcntl import lockf, LOCK_EX, LOCK_UN
from hashlib import md5
from traceback import print_exc, format_exc
from datetime import datetime
from subprocess import Popen,PIPE
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

EMAIL = sys.argv[1]
COLLECTDIR = sys.argv[2] # Directory from where to pick up root files 
TFILEDONEDIR = sys.argv[3] # Directory to store processed *_T files
DROPBOX = sys.argv[4] # Directory where to liave the files

# Constants
WAITTIME = 10
EMAILINTERVAL = 15 * 60 # Time between sent emails 
SBASEDIR = os.path.abspath(__file__).rsplit("/",1)[0]
TMPDROPBOX = "%s/.tmpdropbox" % DROPBOX
RETRIES = 3
SENDMAIL = "/usr/sbin/sendmail" # sendmail location
HOSTNAME = socket.gethostname().lower()

# Control variables
lastEmailSent = datetime.now()

# --------------------------------------------------------------------
def logme(msg, *args):
  procid = "[%s/%d]" % (__file__.rsplit("/", 1)[-1], os.getpid())
  print(datetime.now(), procid, msg % args)
  
def filecheck(rootfile):
  cmd = 'root -l -b -q %s/filechk.C"(\\"%s\\")"' % (SBASEDIR,rootfile)
  a = os.popen(cmd).read().split()
  tag=a.pop()
  if tag == '(int)(-1)' or tag == '(int)0':
    return 0
      
  if tag == '(int)1':
    return 1
  
  return 0
     
def convert(infile, ofile):
  cmd = 'root -l -b -q %s/sistrip_reduce_file.C++"' \
        '(\\"%s\\", \\"%s\\")" >& /dev/null' % (SBASEDIR,infile, ofile)
  os.system(cmd)
  
def sendmail(body="Hello from visDQMZipCastorVerifier"):
  scall = Popen("%s -t" % SENDMAIL, shell=True, stdin=PIPE)
  scall.stdin.write("To: %s\n" % EMAIL)
  scall.stdin.write("Subject: File Collector on server %s has a Critical Error\n" %
                     HOSTNAME)
  scall.stdin.write("\n") # blank line separating headers from body
  scall.stdin.write("%s\n" % body)
  scall.stdin.close()
  rc = scall.wait()
  if rc != 0:
     logme("ERROR: Sendmail exit with status %s", rc)
  
# --------------------------------------------------------------------
if not os.path.exists(TMPDROPBOX):
  os.makedirs(TMPDROPBOX)
  
if not os.path.exists(TFILEDONEDIR):
  os.makedirs(TFILEDONEDIR)
  
if not os.path.exists(DROPBOX):
  os.makedirs(DROPBOX)

while True:
  try:
    NRUNS = 0  #Number of runs found
    NFOUND = 0  #Number of files found
    NEW = {}
    TAGS= []
    for dir, subdirs, files in os.walk(COLLECTDIR):
      for f in files:
        fMatch=re.match('^DQM_V[0-9]{4}_(?P<subsys>.*)_R(?P<runnr>[0-9]{9})(|_T[0-9]*)\.root$',f)
        if not fMatch:
          fMatch=re.match('^Playback_V[0-9]{4}_(?P<subsys>.*)_R(?P<runnr>[0-9]{9})(|_T[0-9]*)\.root$', f)
          
        if fMatch:
          runnr = int(fMatch.group("runnr"))
          subsystem=fMatch.group("subsys")
          runstr="%09d" % runnr
          donefile = "%s/%s/%s/%s" % (TFILEDONEDIR, runstr[0:3], runstr[3:6], f)
          f = "%s/%s" % (dir, f)
          if os.path.exists(donefile) and os.stat(donefile).st_size == os.stat(f).st_size:
            logme("WARNING: File %s was already processed but re-appeared", f)
            os.remove(f)
            continue
            
          NEW.setdefault(runnr, {}).setdefault(subsystem,[]).append(f)
          NFOUND += 1  
          
    if len(NEW) == 0:
      time.sleep(WAITTIME)
      continue
      
    TAGS=sorted(glob.glob('%s/tagfile_runend_*' % COLLECTDIR ),reverse=True)
    if len(TAGS)==0:
      if len(NEW) <= 1:
        time.sleep(WAITTIME)
        continue
        
      TAGRUNEND=int(sorted(NEW.keys(),reverse=True)[1])
      
    else:
      TAGRUNEND=int(TAGS[0].split("_")[2])
      
    for tag in TAGS:
      os.remove(tag)

    for run,subsystems in NEW.items():
      if run > TAGRUNEND:
        continue 
        
      for subsystem,files in  subsystems.items():
        done=False
        keeper=0
        Tfiles=sorted(files,cmp=lambda x,y: "_T" not in x and x != y and 1  or cmp(x,y))[::-1]
        for Tfile in Tfiles:
          seed=HOSTNAME.replace("-","t")[-6:]
          finalTMPfile="%s/DQM_V0001_%s_R%09d.root.%s" % (TMPDROPBOX,subsystem,run,seed)
          runstr="%09d" % run
          finalTfile="%s/%s/%s/%s" % (TFILEDONEDIR,runstr[0:3],runstr[3:6],Tfile.split("/")[-1])
          finalTdir="%s/%s/%s" % (TFILEDONEDIR,runstr[0:3],runstr[3:6])
          if not os.path.exists(finalTdir):
            os.makedirs(finalTdir)
            
          if os.path.exists(finalTMPfile):
            os.remove(finalTMPfile)
          
          if done:
            if keeper == 0:
              keeper+=1
              shutil.move(Tfile,finalTfile+"_d")
              
            else:
              os.remove(Tfile)
              
            continue
                    
          if filecheck(Tfile) != 1:
            logme("INFO: File %s is incomplete looking for next"
                  " DQM_V*_%s_R%09d_T*.root valid file", 
                  Tfile, subsystem, run)
            if keeper == 0:
              keeper+=1
              shutil.move(Tfile,finalTfile+"_d")
              
            else:
              os.remove(Tfile)
              
            continue
            
          if "Playback" in Tfile and "SiStrip" in Tfile:
            dqmfile = Tfile.replace('Playback','DQM')
            convert(Tfile,dqmfile)
            if not os.path.exists(dqmfile):
              logme("WARNING: Problem converting %s skiping", Tfile)
              shutil.move(Tfile,finalTfile+"_d")
              continue
              
            os.rename(Tfile,finalTfile.replace('Playback','Playback_full'))
            Tfile=dqmfile  
            
          for i in range(RETRIES):
            md5Digest=md5(file(Tfile).read())
            originStr="md5:%s %d %s" % (md5Digest.hexdigest(),os.stat(Tfile).st_size,Tfile)
            originTMPFile="%s.origin" % finalTMPfile
            originFile=open(originTMPFile,"w")
            originFile.write(originStr)
            originFile.close() 
            shutil.copy(Tfile,finalTMPfile)
            version=1
            lFile=open("%s/lock" % TMPDROPBOX ,"a")
            lockf(lFile,LOCK_EX)
            for vdir,vsubdir,vfiles in os.walk(DROPBOX):
              if 'DQM_V0001_%s_R%09d.root' % (subsystem,run) not in vfiles:
                continue
              version += 1

            if not os.path.exists("%s/V%04d" % (DROPBOX,version)):
              os.makedirs("%s/V%04d" % (DROPBOX,version))
              
            finalfile="%s/V%04d/DQM_V0001_%s_R%09d.root" %   (DROPBOX,version,subsystem,run)        
            originFileName="%s.origin" % finalfile     
            if os.path.exists(finalTMPfile) and os.stat(finalTMPfile).st_size == os.stat(Tfile).st_size:
              os.rename(Tfile,finalTfile)
              os.rename(finalTMPfile,finalfile)
              os.rename(originTMPFile,originFileName)
              os.chmod(finalfile,stat.S_IREAD|stat.S_IRGRP|stat.S_IROTH| stat.S_IWRITE|stat.S_IWGRP|stat.S_IWOTH)
              os.chmod(originFileName,stat.S_IREAD|stat.S_IRGRP|stat.S_IROTH| stat.S_IWRITE|stat.S_IWGRP|stat.S_IWOTH)  
              logme("INFO: File %s has been successfully sent to the DROPBOX" , Tfile)
              lockf(lFile,LOCK_UN)
              lFile.close()
              break
            else:
              logme("ERROR: Problem transfering final file for run"
                    " %09d. Retrying in %d", run, WAITTIME)
              if i == RETRIES-1: 
                now = datetime.now()
                if now - EMAILINTERVAL > lastEmailSent:
                  sendmail("ERROR: Problem transfering final file for run"
                    " %09d.\n Retrying in %d seconds" % (run, WAITTIME))
                  lastEmailSent = now
                
              time.sleep(WAITTIME)
            lockf(lFile,LOCK_UN)
            lFile.close()
          done=True
          
  except KeyboardInterrupt as e:
    sys.exit(0)

  except Exception as e:
    logme('ERROR: %s', e)
    now = datetime.now()
    if now - EMAILINTERVAL > lastEmailSent:
      sendmail ('ERROR: %s\n%s' % (e, format_exc()))
      lastEmailSent = now
      
    print_exc() 
