#!/usr/bin/env python

import argparse
import subprocess
import socket, fcntl, select, atexit, signal
import sys, os, time, datetime
import collections
import json

def log(s):
    sys.stderr.write("m: " + s + "\n");
    sys.stderr.flush()

def dt2time(dt):
    # convert datetime timstamp to unix
    return time.mktime(dt.timetuple())

class ElasticReport(object):
    def __init__(self, pid, cmdline, history, json):
        self.s_history = history
        self.s_json = json

        self.last_make_report = None
        self.make_report_timer = 30
        self.seq = 0
        
        self.doc = {
            "pid": pid,
            "hostname": socket.gethostname(),
            "sequence": self.seq,
            "cmdline": cmdline,
        }

        self.defaults()

    def defaults(self):
        self.id_format = u"%(type)s-run%(run)06d-host%(hostname)s-pid%(pid)06d"
        self.doc["type"] = "dqm-source-state"
        self.doc["run"] = 0

        # figure out the tag
        c = self.doc["cmdline"]
        for l in c:
            if l.endswith(".py"):
                l = os.path.basename(l)
                l = l.replace(".py", "")
                l = l.replace("_cfg", "")
                self.doc["tag"] = l

        self.make_id()

    def make_id(self):
        id = self.id_format % self.doc
        self.doc["_id"] = id
        return id

    def update_doc(self, keys):
        for key, value in keys.items():
            self.doc[key] = value

    def update_from_json(self):
        while self.s_json.have_docs():
            doc = self.s_json.get_doc()

            # convert some values to integers
            for k in ["pid", "run", "lumi"]:
                if doc.has_key(k):
                    doc[k] = int(doc[k])

            self.update_doc(doc)

    def update_ps_status(self):
        try:
            pid = int(self.doc["pid"])
            fn = "/proc/%d/status" % pid
            f = open(fn, "r")
            d = {}
            for line in f:
                k, v = line.strip().split(":", 1)
                d[k.strip()] = v.strip()
            f.close()

            self.doc["ps_info"] = d
        except:
            pass

    def update_stderr(self):
        if self.s_history:
            self.doc["stderr"] = self.s_history.read()

    def make_report(self):
        self.last_make_report = time.time()
        self.doc["report_timestamp"] = time.time()

        self.update_from_json()
        self.make_id()
        self.update_ps_status()
        self.update_stderr()
        #print self.doc

        fn_id = self.doc["_id"] + ".jsn"
        fn = os.path.join("/tmp/dqm_monitoring/", fn_id) 
        fn_tmp = os.path.join("/tmp/dqm_monitoring/", fn_id + ".tmp") 

        with open(fn_tmp, "w") as f:
            json.dump(self.doc, f, indent=True)

        os.rename(fn_tmp, fn)

    def try_update(self):
        # first time
        if self.last_make_report is None:
            return self.make_report()

        # is json stream has updates
        if self.s_json and self.s_json.have_docs():
            return self.make_report()

        now = time.time()
        delta = now - self.last_make_report
        if delta > self.make_report_timer:
            return self.make_report()

    def write(self, rbuf):
        self.try_update()

    def flush(self):
        self.try_update()

class History(object):
    def __init__(self, history_size=64*1024):
        self.max_size = history_size
        self.buf = collections.deque()
        self.size = 0

    def pop(self):
        if not len(self.buf):
            return None

        elm = self.buf.popleft()
        self.size -= len(elm)

        return elm

    def push(self, rbuf):
        self.buf.append(rbuf)
        self.size += len(rbuf)

    def write(self, rbuf):
        l = len(rbuf)
        while (self.size + l) >= self.max_size:
            self.pop()

        self.push(rbuf)

    def read(self):
        return "".join(self.buf)

    def flush(self):
        pass

class JsonInput(object):
    def __init__(self):
        self.buf = []
        self.docs = []
    
    def parse_line(self, line):
        if not line.strip():
            # this is keep alive
            # not yet implemented
            return

        try:
            doc = json.loads(line)
            self.docs.append(doc)
        except:
            log("cannot deserialize json: %s" % line)

    def get_doc(self):
        return self.docs.pop(0)

    def have_docs(self):
        return len(self.docs) > 0

    def write(self, rbuf):
        self.buf.append(rbuf)
        if "\n" in rbuf:
            # split whatever we have
            all = "".join(self.buf)
            spl = all.split("\n")

            while len(spl) > 1:
                line = spl.pop(0)
                self.parse_line(line)

            self.buf = [spl[0]]

    def flush(self):
        pass

class DescriptorCapture(object):
    def __init__(self, f, write_files=[]):
        self.f = f
        self.fd = f.fileno()
        self.write_files = write_files

    def read_in(self, rbuf):
        for f in self.write_files:
            f.write(rbuf)
            f.flush()

    def close_in(self):
        log("closed fd %d" % self.fd)
        self.f.close()

    @staticmethod
    def event_loop(desc, timeout, timeout_call=None):
        fd_map = {}
        p = select.poll()

        for desc in desc:
            fd_map[desc.fd] = desc
            p.register(desc.fd, select.POLLIN)

        while len(fd_map) > 0:
            events = p.poll(timeout)
            if len(events) == 0:
                if timeout_call:
                    timeout_call()

            for fd, ev in events:
                rbuf = os.read(fd, 1024)
                if len(rbuf) == 0:
                    fd_map[fd].close_in()

                    p.unregister(fd)
                    del fd_map[fd]
                else:
                    fd_map[fd].read_in(rbuf)


def create_fifo():
    prefix = "/tmp"
    if os.path.isdir("/tmp/dqm_monitoring"):
        prefix = "/tmp/dqm_monitoring"

    base = ".es_monitoring_pid%08d" % os.getpid()
    fn = os.path.join(prefix, base)

    if os.path.exists(fn):
        os.unlink(fn)

    os.mkfifo(fn, 0600)
    if not os.path.exists(fn):
        log("Failed to create fifo file: %s" % fn)
        sys.exit(-1)

    atexit.register(os.unlink, fn)
    return fn

CURRENT_PROC = []
def launch_monitoring(args):
    fifo = create_fifo()
    mon_fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)

    def preexec():
        # this should only be open on a parent
        os.close(mon_fd)

        # open fifo once (hack)
        # so there is *always* at least one writter
        # which closes with the executable
        os.open(fifo, os.O_WRONLY)

        try:
            # ensure the child dies if we are SIGKILLED
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        except:
            log("Failed to setup PR_SET_PDEATHSIG.")
            pass

        env = os.environ
        env["DQMMON_UPDATE_PIPE"] = fifo

    p = subprocess.Popen(args.pargs, preexec_fn=preexec, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    CURRENT_PROC.append(p)

    mon_file = os.fdopen(mon_fd)
    s_hist = History()
    s_json = JsonInput()
    report_sink = ElasticReport(pid=p.pid, cmdline=args.pargs, history=s_hist, json=s_json)

    stdout_cap = DescriptorCapture(p.stdout, write_files=[sys.stdout, s_hist, report_sink, ], )
    stderr_cap = DescriptorCapture(p.stderr, write_files=[sys.stderr, s_hist, report_sink, ], )
    stdmon_cap = DescriptorCapture(mon_file, write_files=[s_json, report_sink, ],)

    fs = [stdout_cap, stderr_cap, stdmon_cap]
    try:
        DescriptorCapture.event_loop(fs, timeout=1000, timeout_call=report_sink.flush)
    except select.error, e:
        # we have this on ctrl+c
        # just terminate the child
        log("Select error (we will terminate): " + str(e))
        p.terminate()

    # at this point the program is dead
    r =  p.wait()
    CURRENT_PROC.remove(p)

    report_sink.update_doc({ "exit_code": r })
    report_sink.make_report()

    return r

def handle_signal(signum, frame):
    for proc in CURRENT_PROC:
        proc.send_signal(signum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kill/restart the child process if it doesn't out the required string.")
    parser.add_argument("-t", type=int, default="2", help="Timeout in seconds.")
    parser.add_argument("-s", type=int, default="2000", help="Signal to send.")
    parser.add_argument("-r", "--restart",  action="store_true", default=False, help="Restart the process after killing it.")
    parser.add_argument("pargs", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # do some signal magic
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    sys.exit(launch_monitoring(args))
