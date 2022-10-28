#! /usr/bin/env python3

from __future__ import print_function
import os,time,sys,shutil,glob
from sets import Set
from datetime import datetime
import smtplib
from email.MIMEText import MIMEText
from ROOT import TFile

def sendmail(EmailAddress,run):
    s=smtplib.SMTP("localhost")
    tolist=[EmailAddress]
    body="File merge failed by unknown reason for run"+run
    msg = MIMEText(body)
    msg['Subject'] = "File merge failed."
    msg['From'] = ServerMail
    msg['To'] = EmailAddress
    s.sendmail(ServerMail,tolist,msg.as_string())
    s.quit()

def filecheck(rootfile):
    f = TFile(rootfile)
    if (f.IsZombie()):
        #print "File corrupted"
        f.Close()
        return 0
    else:
        hist = f.FindObjectAny("reportSummaryContents")
        #(skip filecheck for HcalTiming files!!)
        if (hist == None and rootfile.rfind('HcalTiming') == -1):
            #print "File is incomplete"
            f.Close()
            return 0
        else:
            #print "File is OK"
            f.Close()
            return 1



TempTag = TimeTag + '-tmp'
if not os.path.exists(TimeTag):
    os.system('touch -t 01010000 '+ TimeTag)


#### search new files
NFOUND = 0
NEW = {}
TAG = os.stat(TimeTag).st_mtime
for dir, subdirs, files in os.walk(DIR):
    paths = ["%s/%s" % (dir, x) for x in files]
    for f in [x for x in paths if re.search(r'/DQM_.*_R[0-9]{9}\.root$', x)]:
        if os.stat(f).st_mtime > TAG and f.index("DQM_Reference") < 0:
            NEW.get(f[-14:-5], []).append(f)
            NFOUND += 1

if not NFOUND:
    print('Have not found new files...')
    os._exit(99)

os.system("ls -l %s" % TimeTag)
os.system('touch '+ TempTag)
print(datetime.now())
print('Found %d new file(s).' % NFOUND)

#### loop for runs
newFiles = []
allOldFiles = []
for (run, files) in NEW.items():
    runnr = "%09d" % long(run)
    destdir = "%s/%s/%s/%s" % (FILEDIR, runnr[0:3], runnr[3:6], runnr[6:9])
    oldfiles = []
    version = 1
    while True:
        destfile = "%s/DQM_V%04d_R%s.root" % (destdir, version, runnr)
        if not os.path.exists(destfile): break
        oldfiles.append(destfile)
        version += 1

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    logfile = "%s.log" % destfile[:-4]
    tmpdestfile = "%s.tmp" % destfile

    print('Run %s is being merged...' % run)

    if os.path.exists(tmpdestfile):
        os.remove(tmpdestfile)

    ntries = 0
    while ntries < 30:
        LOGFILE = file(logfile, 'a')
        LOGFILE.write(os.popen('DQMMergeFile %s %s' % (tmpdestfile, " ".join(files))).read())
        if not os.path.exists(tmpdestfile):
            print('Failed merging files for run %s. Try again after two minutes' % run)
            time.sleep(WAITTIME)
            ntries += 1
        else:
            break

    os.rename(tmpdestfile, destfile)
    sendmail(YourEmail,run)
    allOldFiles.extend(oldfiles)
    newFiles.append(destfile)


if os.path.exists(TMPDB):
    os.remove(TMPDB)

if os.path.exists(DB):
    os.rename(DB, TMPDB)
else:
    logfile.write('*** INITIALISE DATABASE ***\n')
    logfile.write(os.popen('visDQMRegisterFile %s "/Global/Online/ALL" "Global run"' % TMPDB).read())

logfile.write('*** UNREGISTER %d OLD FILES ***\n' % len(allOldFiles))
while len(allOldFiles) > 0:
    (slice, rest) = (allOldFiles[0:50], allOldFiles[50:])
    logfile.write(os.popen('visDQMUnregisterFile %s %s' % (tmpdb, " ".join(slice))).read())
    allOldFiles = rest
    for old in slice:
        os.remove(old)

for file in newFiles:
    print('Registering %s' % file)
    logfile.write('*** REGISTER FILE %s ***\n' % file)
    logfile.write(os.popen('visDQMRegisterFile %s "/Global/Online/ALL" "Global run" %s' % (TMPDB, file)).read())
    print('%s registered' % file)

os.rename(TMPDB, DB)
os.remove(TimeTag)
os.rename(TempTag, TimeTag)
