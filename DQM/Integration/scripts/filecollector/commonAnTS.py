import os, datetime, time,  sys, shutil, glob, re, subprocess as sp, tempfile, socket
TIME_OUT=700
DEBUG=False
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

def getDiskUsage(path):
  fsStats=os.statvfs(path)
  size=fsStats.f_bsize*fsStats.f_blocks
  available=fsStats.f_bavail*fsStats.f_bsize
  used=size-available
  usedPer=float(used)/size
  return (size,available,used,usedPer)

def getNumRunsWithinTime(path,timeDelta):
  numRuns=0
  currTime=time.time()
  STAY_DIR={}
  for directory,subdirs,files in os.walk(path):
    for f in sorted(files, key=lambda x: os.stat("%s/%s" % (directory,x)).st_mtime,reverse=True):
      fullFName="%s/%s" % (directory,f)
      fMatch=re.match(r".*_R([0-9]{9}).*",f)
      if fMatch:
        run=fMatch.group(1)
        fMtime=os.stat(fullFName).st_mtime
        if currTime - timeDelta*3600 <= fMtime:
          STAY_DIR.setdefault(run,fMtime)
        else:
          break
  numRuns=len(STAY_DIR)
  return numRuns
  
def debugMsg(level,message):
  LEVELS=["INFO","WARNING","ERROR"]
  d=datetime.datetime.today()
  timeStamp=d.strftime("%Y/%m/%d\t%H:%M:%S")
  msg="%s\t%s:\t%s\n" % (timeStamp,LEVELS[level],message)
  sys.stdout.write(msg)
  return True
  
def prettyPrintUnits(value,unit,decimals=0):
  """
Provide human readable units
  """
  runit=""
  if unit is "b":
    units=["B","KB","MB","GB","TB"]
    it=iter(units)
    v=int(value/1024)
    p=0
    runit=next(it)
    while v > 0:
      v=int(v/1024)
      try:
        runit=next(it)
        p+=1
      except:
        break
    return "%%.%df %%s " % decimals % (float(value)/pow(1024,p),runit)
  else:
    return "%%.%df %%s " % decimals % (value,"%") 
  
def executeCmd(cmd):
  stdOutFile=tempfile.TemporaryFile(bufsize=0)
  stdErrFile=tempfile.TemporaryFile(bufsize=0)
  cmdHdl=sp.Popen(cmd,shell=True,stdout=stdOutFile,stderr=stdErrFile)
  t=0
  cmdHdl.poll()
  while cmdHdl.returncode == None and t<TIME_OUT:
    t=t+1
    cmdHdl.poll()
    time.sleep(1)
  if t >= TIME_OUT and not cmdHdl.returncode:
    try:
      os.kill(cmdHdl.pid,9)
      debugMsg(2,"Execution timed out on Command: '%s'" % cmd)
    except:
      DEBUG and debugMsg(1,"Execution timed out on Command: '%s' but it ended while trying to kill it, adjust timer" % cmd)
  cmdHdl.poll()
  DEBUG and debugMsg(0,"End of Execution cicle of Command: '%s' " % cmd)
  stdOutFile.seek(0)
  stdErrFile.seek(0)
  return (stdOutFile,stdErrFile,cmdHdl.returncode)
  
  
def sendmail(EmailAddress,run=123456789,body="",subject="File merge failed."):
  import os, smtplib
  from email.MIMEText import MIMEText
  server=socket.gethostname() #os.getenv("HOSTNAME")
  user=os.getenv("USER")
  ServerMail="%s@%s" % (user,server)
  s=smtplib.SMTP("localhost")
  tolist=[EmailAddress] #[EmailAddress, "lat@cern.ch"]
  if not body: body="File copy to dropbox failed by unknown reason for run:%09d on server: %s" % (run,server)
  msg = MIMEText(body)
  msg['Subject'] = subject
  msg['From'] = ServerMail
  msg['To'] = EmailAddress
  s.sendmail(ServerMail,tolist,msg.as_string())
  s.quit()
