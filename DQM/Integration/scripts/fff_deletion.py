import os
import logging
import re
import datetime
import subprocess
import socket
import time

logging.basicConfig(level=logging.INFO)
log = logging

re_files = re.compile(r"^run(?P<run>\d+)/run(?P<runf>\d+)_ls(?P<ls>\d+)_.+\.(dat|raw)+(\.deleted)*")
def parse_file_name(rl):
    m = re_files.match(rl)
    if not m:
        return None

    d = m.groupdict()
    sort_key = (int(d["run"]), int(d["runf"]), int(d["ls"]), )
    return sort_key

def iterate(top, stopSize, action):
    # entry format (path, size)
    collected = [] 

    for root, dirs, files in os.walk(top, topdown=True):
        for name in files:
            fp = os.path.join(root, name)
            rl = os.path.relpath(fp, top)

            sort_key = parse_file_name(rl)
            if sort_key:
                fsize = os.stat(fp).st_size
                if fsize == 0:
                    continue

                sort_key = parse_file_name(rl)
                collected.append((sort_key, fp, fsize, ))

    # for now just use simple sort
    collected.sort(key=lambda x: x[0])

    # do the action
    for sort_key, fp, fsize in collected:
        if stopSize <= 0:
            break

        action(fp)
        stopSize = stopSize - fsize

def cleanup_threshold(top, threshold, action, string):
    st = os.statvfs(top)
    total = st.f_blocks * st.f_frsize
    used = total - (st.f_bavail * st.f_frsize)
    threshold = used - float(total * threshold) / 100

    def p(x):
        return float(x) * 100 / total

    log.info("Using %d (%.02f%%) of %d space, %d (%.02f%%) above %s threshold.",
        used, p(used), total, threshold, p(threshold), string)

    if threshold > 0:
        iterate(top, threshold, action)
        log.info("Done cleaning up for %s threshold.", string)
    else:
        log.info("Threshold %s not reached, doing nothing.", string)

def diskusage(top):
    st = os.statvfs(top)
    total = st.f_blocks * st.f_frsize
    used = total - (st.f_bavail * st.f_frsize)
    return float(used) * 100 / total

class FileDeleter(object):
    def __init__(self, top, thresholds, email_to, fake=True, ):
        self.top = top
        self.fake = fake
        self.email_to = email_to
        self.thresholds = thresholds

        self.last_email = None
        self.min_interval = datetime.timedelta(seconds=60*10)
        self.hostname = socket.gethostname()

    def rename(self, f):
        if f.endswith(".deleted"):
            return

        fn = f + ".deleted"

        if self.fake:
            log.warning("Renaming file (fake): %s -> %s", f, 
                os.path.relpath(fn, os.path.dirname(f)))
        else:
            log.warning("Renaming file: %s -> %s", f, 
                os.path.relpath(fn, os.path.dirname(f)))

            os.rename(f, fn)

    def delete(self, f):
        if not f.endswith(".deleted"):
            return
    
        if self.fake:
            log.warning("Truncating file (fake): %s", f)
        else:
            log.warning("Truncating file: %s", f)
            open(f, "w").close()

    def send_smg(self, used_pc):
        now = datetime.datetime.now()

        if (self.last_email is not None):
            if (now - self.last_email) < self.min_interval:
                return

        self.last_email = now

        # sms service does not accept an email with a several recipients
        # so we send one-by-one
        for email in self.email_to:
            subject = "Disk out of space (%.02f%%) on %s." % (used_pc, self.hostname)
            if "mail2sms" in email:
                text = ""
            else:
                text = subject

            log.info("Sending email: %s", repr(["/bin/mail", "-s", subject, email]))
            p = subprocess.Popen(["/bin/mail", "-s", subject, email], stdin=subprocess.PIPE, shell=False)
            p.communicate(input=text)

    def run(self):            
        cleanup_threshold(self.top, self.thresholds['rename'], self.rename, "rename")
        cleanup_threshold(self.top, self.thresholds['delete'], self.delete, "delete")
   
        du = diskusage(self.top)
        if du > self.thresholds['email']: 
            deleter.send_smg(du)

# use a named socket check if we are running
# this is very clean and atomic and leave no files
# from: http://stackoverflow.com/a/7758075
def lock(pname):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.bind('\0' + pname)
        return sock
    except socket.error:
        return None

def daemon(deleter, delay_seconds=30):
    while True:
        deleter.run()
        time.sleep(delay_seconds)
   
import sys 
if __name__ == "__main__":
    #import argparse
    #parser = argparse.ArgumentParser(description="Delete files if disk space usage reaches critical level.")
    #parser.add_argument("-r", "--renameT", type=float, help="Percentage of total disk space used for file renaming.")
    #parser.add_argument("-d", "--deleteT", type=float, help="Percentage of total disk space used for file deletion.")
    #parser.add_argument("-t", "--top", type=str, help="Top level directory.", default="/fff/ramdisk/")
    #args = parser.parse_args()

    # try to take the lock or quit
    sock = lock("fff_deleter")
    if sock is None:
        log.info("Already running, exitting.")
        sys.exit(0)

    # threshold rename and delete must be in order
    # in other words, always: delete > rename
    # this is because delete only deletes renamed files

    # email threshold has no restrictions
    top = "/fff.changeme/ramdisk"
    thresholds = {
        'delete': 80,
        'rename': 60,
        'email':  90,
    }
    fake = not (len(sys.argv) > 1 and sys.argv[1] == "doit")

    deleter = FileDeleter(
        top = top,
        thresholds = thresholds,
        # put "41XXXXXXXXX@mail2sms.cern.ch" to send the sms
        email_to = ["dmitrijus.bugelskis@cern.ch", ],
        fake = fake,
    )

    daemon(deleter=deleter) 
