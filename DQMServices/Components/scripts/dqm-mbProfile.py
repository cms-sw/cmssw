#!/usr/bin/env python3

import os
import collections
import logging
import resource
import time
import argparse
import subprocess
import signal
import json
import inspect
import shutil

LOG_FORMAT='%(asctime)s: %(name)-20s - %(levelname)-8s - %(message)s'
logging.basicConfig(format=LOG_FORMAT)
log = logging.getLogger("mbProfile")
log.setLevel(logging.INFO)

def read_procfs(ppath, only_ppid=True):
    def read(f):
        fp = os.path.join(ppath, f)
        with open(fp) as fd:
            return fd.read()

    def read_status():
        st = {}

        fp = os.path.join(ppath, "status")
        with open(fp) as fd:
            for line in fd.readlines():
                if not line: continue

                key, value = line.split(":", 1)
                st[key] = value.strip()

        return st

    try:
        dct = {}

        dct["statm"] = read("statm").strip()
        dct["stat"] = read("stat").strip()
        dct["cmdline"] = read("cmdline").strip().replace("\0", " ")

        status = read_status()
        dct["status"] = status
        dct["pid"] = int(status["Pid"])
        dct["parent_pid"] = int(status["PPid"])

        return dct
    except:
        log.warning("Exception in read_procfs.", exc_info=True)
        pass

def build_process_list():
    lst = os.listdir("/proc/")
    for f in lst:
        if not f.isdigit(): continue

        proc = read_procfs(os.path.join("/proc", f))
        if proc:
            yield proc

def get_children(ppid):
    """ Select all processes which are descendant from ppid (exclusive). """

    pid_dct = {}
    for proc in build_process_list():
        proc["_children"] = []
        pid_dct[proc["pid"]] = proc

    # fill in children array
    for pid in list(pid_dct.keys()):
        parent_pid = pid_dct[pid]["parent_pid"]

        if parent_pid in pid_dct:
            pid_dct[parent_pid]["_children"].append(pid)

    # now just walk down the tree
    if ppid is None or ppid not in pid_dct:
        # process has quit, we exit
        return []

    accepted = []
    to_accept = collections.deque([ppid, ])
    
    while to_accept:
        head = pid_dct[to_accept.popleft()]

        # do not include the monitoring pid
        if head["pid"] != ppid:
            accepted.append(head)

        to_accept.extend(head.get("_children", []))
        head["children"] = head["_children"]
        del head["_children"]

        # deleting children breaks infinite loops
        # but Dima, can a process tree contain a loop? yes - via race-condition in reading procfs

    return accepted

class Profile(object):
    def __init__(self, args):
        self.time = time.time()
        self.final = False
        self.pid = None 
        self.known_pids = {}

        self.ru = {}
        self.ru_diff = {}

        self._offset_ru = None
        self._args = args

        if self._args.file:
            self._file = open(self._args.file, "w")
        else:
            self._file = None

        self.update()

    def update_ru(self):
        fields_to_subtract = (
            "ru_utime", "ru_stime", "ru_maxrss", "ru_minflt", "ru_majflt", "ru_nswap",
            "ru_inblock", "ru_oublock", "ru_msgsnd", "ru_msgrcv", "ru_nsignals", "ru_nvcsw", "ru_nivcsw",
        )

        rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.ru = rusage

        if self._offset_ru is None:
            self._offset_ru = rusage

        for field in fields_to_subtract:
            current = getattr(self.ru, field)
            base = getattr(self._offset_ru, field)

            self.ru_diff[field] = current - base

    # this is taken from: http://github.com/pixelb/scripts/commits/master/scripts/ps_mem.py
    def read_smaps(self, proc_dict):
        Private, Shared, Pss = 0, 0, 0
 
        fp = os.path.join("/proc/%d" % proc_dict["pid"], "smaps")
        with open(fp) as fd:
            for line in fd.readlines():
                if line.startswith("Shared"):
                    Shared += int(line.split()[1])
                elif line.startswith("Private"):
                    Private += int(line.split()[1])
                elif line.startswith("Pss"):
                    Pss += int(line.split()[1])
    
        proc_dict["smaps_shared"] = Shared * 1024
        proc_dict["smaps_private"] = Private * 1024
        proc_dict["smaps_pss"] = Pss * 1024

    def update_proc(self):
        procs = get_children(os.getpid())

        # we can only do it here, permision-wise
        # ie only for owned processes
        for proc in procs:
            try:
                self.read_smaps(proc)
            except:
                log.warning("Exception in read_smaps.", exc_info=True)

        # we need to mark not-running ones as such
        stopped = set(self.known_pids.keys())
        for proc in procs:
            proc["running"] = True

            pid = proc["pid"]
            self.known_pids[pid] = proc

            if pid in stopped:
                stopped.remove(pid)

        for pid in stopped:
            self.known_pids[pid]["running"] = False

    def update(self):
        self.time = time.time()

        self.update_ru()
        self.update_proc()

        if self._file:
            json.dump(self.to_dict(), self._file)
            self._file.write("\n")
            self._file.flush()

        log.info("Written profile to: %s, took=%.03f", self._args.file, time.time() - self.time)

    def to_dict(self):
        dct = collections.OrderedDict()
        dct['time']         = self.time
        dct['pid']          = self.pid
        dct['final']        = self.final
        
        dct['ru_diff']      = dict(self.ru_diff)
        dct['ru']           = dict((k, v) for k, v in inspect.getmembers(self.ru) if k.startswith('ru_'))
        dct['known_pids']   = dict(self.known_pids)
        return dct
    
    def finish(self):
        self.final = True
        self.update()

        if self._file:
            self._file.close()
            self._file = None
        else:
            log.info("ru_diff: %s", self.ru_diff)


ALARM_TIMER = 1
ALARM_P_OBJECT = None

def handle_alarm(num, frame):
    if ALARM_P_OBJECT:
        ALARM_P_OBJECT.update()

    signal.alarm(ALARM_TIMER)

def run_and_monitor(args):
    profile = Profile(args)

    proc = subprocess.Popen(args.pargs)
    profile.pid = proc.pid

    global ALARM_P_OBJECT
    ALARM_P_OBJECT = profile

    signal.signal(signal.SIGALRM, handle_alarm)
    signal.alarm(ALARM_TIMER)

    proc.wait()
    profile.finish()

def find_and_write_html(p, args):
    # create the dir if necessary
    if p and not os.path.exists(p):
        os.makedirs(p)

    html_paths = [
        os.path.join(os.getenv("CMSSW_BASE"),"src/DQMServices/Components/data/html"),
        os.path.join(os.getenv("CMSSW_RELEASE_BASE"), "src/DQMServices/Components/data/html"),
    ]

    def find_file(f):
        fails = []
        for p in html_paths:
            x = os.path.join(p, f)
            if os.path.exists(x):
                return x
            else:
                fails.append(x)

        log.warning("Could not find html file: %s (%s)", f, fails)

    for f in ['mbGraph.js', 'mbGraph.html']:
        target_fn = os.path.join(p, f)
        source_fn = find_file(f)
        if source_fn:
            log.info("Copying %s to %s", source_fn, target_fn)
            shutil.copyfile(source_fn, target_fn)

    # create json file
    target_fn = os.path.join(p, "mbGraph.json")
    log.info("Creating %s", target_fn)
    with open(target_fn, "w") as fp:
        dct = {
            "file": os.path.basename(args.file),
            "interval": args.i,
            "env": {
                "CMSSW_GIT_HASH": os.getenv("CMSSW_GIT_HASH"),
                "CMSSW_RELEASE_BASE": os.getenv("CMSSW_RELEASE_BASE"),
                "SCRAM_ARCH": os.getenv("SCRAM_ARCH"),
            },
        }

        json.dump(dct, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile child processes and produce data for rss and such graphs.")
    parser.add_argument("-f", "--file", type=str, default="performance.json", help="Filename to write.", metavar="performance.json")
    parser.add_argument("-i", type=int, help="Time interval between profiles.", default=15)
    parser.add_argument('-q', action='store_true', help="Reduce logging.")
    parser.add_argument('-w', action='store_true', help="Write html helper files for rendering the performance file.")
    parser.add_argument('pargs', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if not args.pargs:
        parser.print_help()
        sys.exit(-1)
    elif args.pargs[0] == "--":
        # compat with 2.6
        args.pargs = args.pargs[1:]

    ALARM_TIMER = args.i

    if args.q:
        log.setLevel(logging.WARNING)

    if args.w:
        p = os.path.dirname(args.file)
        find_and_write_html(p, args)

    ## do some signal magic
    #signal.signal(signal.SIGINT, handle_signal)
    #signal.signal(signal.SIGTERM, handle_signal)

    run_and_monitor(args)

