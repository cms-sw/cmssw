#!/usr/bin/env python3

import argparse
import subprocess
import socket, fcntl, select, atexit, signal, asyncore
import sys, os, time, datetime
import collections
import json
import zlib

def log(s):
    sys.stderr.write("m: " + s + "\n");
    sys.stderr.flush()

def dt2time(dt):
    # convert datetime timstamp to unix
    return time.mktime(dt.timetuple())

class JsonEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'to_json'):
                return obj.to_json()

            return json.JSONEncoder.default(self, obj)

class ElasticReport(object):
    def __init__(self, args):
        self.last_make_report = None
        self.make_report_timer = 30
        self.seq = 0
        self.args = args
        
        self.doc = {
            "hostname": socket.gethostname(),
            "sequence": self.seq,
            "cmdline": args.pargs,
        }

    def defaults(self):
        self.id_format = u"%(type)s-run%(run)06d-host%(hostname)s-pid%(pid)06d"
        self.doc["type"] = "dqm-source-state"
        self.doc["run"] = 0

        # figure out the tag
        c = self.doc["cmdline"]
        for l in c:
            if l.endswith(".py"):
                t = os.path.basename(l)
                t = t.replace(".py", "")
                t = t.replace("_cfg", "")
                self.doc["tag"] = t

            pr = l.split("=")
            if len(pr) > 1 and pr[0] == "runNumber" and pr[1].isdigit():
                run = int(pr[1])
                self.doc["run"] = run

        self.make_id()

        #if os.environ.has_key("GZIP_LOG"):
        #    self.doc["stdlog_gzip"] = os.environ["GZIP_LOG"]

        try:
            self.doc["stdout_fn"] = os.readlink("/proc/self/fd/1")
            self.doc["stderr_fn"] = os.readlink("/proc/self/fd/2")
        except:
            pass

        self.update_doc({ "extra": {
            "environ": dict(os.environ)
        }})

    def make_id(self):
        id = self.id_format % self.doc
        self.doc["_id"] = id
        return id

    def update_doc_recursive(self, old_obj, new_obj):
        for key, value in new_obj.items():
            if (key in old_obj and 
                isinstance(value, dict) and 
                isinstance(old_obj[key], dict)):

                self.update_doc_recursive(old_obj[key], value)
            else:
                old_obj[key] = value

    def update_doc(self, keys):
        self.update_doc_recursive(self.doc, keys)

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

            self.update_doc({ 'extra': { 'ps_info': d } })
        except:
            pass

    def update_mem_status(self):
        try:
            key = str(time.time())

            pid = int(self.doc["pid"])
            fn = "/proc/%d/statm" % pid
            f = open(fn, "r")
            dct = { key: f.read().strip() }
            f.close()

            self.update_doc({ 'extra': { 'mem_info': dct } })
        except:
            pass

    def make_report(self):
        self.last_make_report = time.time()
        self.doc["report_timestamp"] = time.time()
        self.make_id()

        m_path = self.args.path

        if not os.path.isdir(m_path):
            if self.args.debug:
                log("File not written, because report directory does not exist: %s." % m_path)
            # don't make a report if the directory is not available
            return

        self.update_ps_status()
        self.update_mem_status()

        fn_id = self.doc["_id"] + ".jsn"

        fn = os.path.join(m_path, fn_id) 
        fn_tmp = os.path.join(m_path, fn_id + ".tmp") 

        with open(fn_tmp, "w") as f:
            json.dump(self.doc, f, indent=True, cls=JsonEncoder)

        os.rename(fn_tmp, fn)

        if self.args.debug:
            log("File %s written." % fn)

    def try_update(self):
        # first time
        if self.last_make_report is None:
            return self.make_report()

        now = time.time()
        delta = now - self.last_make_report
        if delta > self.make_report_timer:
            return self.make_report()

class LineHistoryEnd(object):
    def __init__(self, max_bytes=16*1024, max_lines=256):
        self.max_bytes = max_bytes
        self.max_lines = max_lines

        self.buf = collections.deque()
        self.size = 0

    def pop(self):
        elm = self.buf.popleft()
        self.size -= len(elm)

    def push(self, rbuf):
        self.buf.append(rbuf)
        self.size += len(rbuf)

    def write(self, line):
        line_size = len(line)

        while len(self.buf) and ((self.size + line_size) > self.max_bytes):
            self.pop()

        while (len(self.buf) + 1) > self.max_lines:
            self.pop()

        self.push(line)

    def to_json(self):
        return list(self.buf)

class LineHistoryStart(LineHistoryEnd):
    def __init__(self, *kargs, **kwargs):
        LineHistoryEnd.__init__(self, *kargs, **kwargs)
        self.done = False

    def write(self, line):
        if self.done:
            return

        if ((self.size + len(line)) > self.max_bytes):
            self.done = True
            return

        if (len(self.buf) > self.max_lines):
            self.done = True
            return

        self.push(line)

class AsyncLineReaderMixin(object):
    def __init__(self):
        self.line_buf = []

    def handle_close(self):
        # closing fd
        if len(self.line_buf):
            self.handle_line("".join(self.line_buf))
            self.line_buf = []

        self.close()

    def handle_read(self):
        rbuf = self.recv(1024*16)
        rbuf = rbuf.decode('utf-8')
        ## not needed, since asyncore automatically handles close
        #if len(rbuf) == 0:
        #    self.handle_close()
        #    return

        self.line_buf.append(rbuf)
        if "\n" in rbuf:
            # split whatever we have
            spl = "".join(self.line_buf).split("\n")

            while len(spl) > 1:
                line = spl.pop(0)
                self.handle_line(line + "\n")

            if len(spl[0]):
                self.line_buf = [spl[0]]
            else:
                self.line_buf = []

    def handle_line(self):
        # override this!
        pass

class AsyncLineReaderTimeoutMixin(AsyncLineReaderMixin):
    def __init__(self, timeout_secs):
        self.timeout_secs = timeout_secs
        self.last_read = time.time()

        super(AsyncLineReaderTimeoutMixin, self).__init__()

    def handle_read(self):
        self.last_read = time.time()
        AsyncLineReaderMixin.handle_read(self)

    def readable(self):
        if (time.time() - self.last_read) >= self.timeout_secs:
            self.last_read = time.time()
            self.handle_timeout()

        return super(AsyncLineReaderTimeoutMixin, self).readable()

class FDJsonHandler(AsyncLineReaderMixin, asyncore.dispatcher):
    def __init__(self, sock, es):
        AsyncLineReaderMixin.__init__(self)
        asyncore.dispatcher.__init__(self, sock)

        self.es = es

    def handle_line(self, line):
        if len(line) < 4:
            # keep alive 'ping'
            self.es.try_update()
            return

        try:
            doc = json.loads(line)

            for k in ["pid", "run", "lumi"]:
                if k in doc:
                    doc[k] = int(doc[k])

            self.es.update_doc_recursive(self.es.doc, doc)
            self.es.try_update()
        except:
            log("cannot deserialize json len: %d content: %s" % (len(line), line))

    def handle_write(self):
        pass

    def writable(self):
        return False

class FDJsonServer(asyncore.file_dispatcher):
    def __init__(self, es, args):
        asyncore.dispatcher.__init__(self)

        self.fn = None
        self.es = es
        self.args = args

        prefix = "/tmp"
        if os.path.isdir(self.args.path):
            prefix = self.args.path

        base = ".es_monitoring_pid%08d" % os.getpid()
        self.fn = os.path.join(prefix, base)

        if self.args.debug:
            log("Socket path: %s" % self.fn)

        if os.path.exists(self.fn):
            os.unlink(self.fn)

        self.create_socket(socket.AF_UNIX, socket.SOCK_STREAM)
        oldmask = os.umask(0o077)
        try:
            self.bind(self.fn)
            self.listen(5)
        finally:
            os.umask(oldmask)
            pass

        atexit.register(self.cleanup)

    def cleanup(self):
        if self.fn is not None:
            if os.path.exists(self.fn):
                os.unlink(self.fn)

    def handle_accept(self):
        pair = self.accept()
        if pair is not None:
            handler = FDJsonHandler(pair[0], self.es)

    def handle_close(self):
        self.close()
        self.cleanup()

class FDOutputListener(AsyncLineReaderTimeoutMixin, asyncore.file_dispatcher):
    def __init__(self, fd, es, zlog, close_socket=None):
        AsyncLineReaderTimeoutMixin.__init__(self, 5)
        asyncore.file_dispatcher.__init__(self, fd)

        self.es = es
        self.zlog = zlog
        self.close_socket = close_socket

        self.start = LineHistoryStart();
        self.end = LineHistoryEnd()

        self.es.update_doc({ 'extra': { 'stdlog_start': self.start } })
        self.es.update_doc({ 'extra': { 'stdlog_end': self.end } })

    def writable(self):
        return False

    def handle_line(self, line):
        if self.zlog is not None:
            self.zlog.write(line)
        else:
            sys.stdout.write(line)
            sys.stdout.flush()
        
        self.start.write(line)
        self.end.write(line)
        self.es.try_update()

    def handle_timeout(self):
        self.es.try_update()

        if self.zlog is not None:
            self.zlog.handle_timeout()

    def handle_close(self):
        super(FDOutputListener, self).handle_close()

        if self.close_socket is not None:
            self.close_socket.handle_close()
    
    def finish(self):
        if self.zlog is not None:
            self.zlog.finish()


CURRENT_PROC = []
def launch_monitoring(args):
    es = ElasticReport(args=args)

    json_handler = FDJsonServer(es=es, args=args)
    env = os.environ.copy()
    env["DQM2_SOCKET"] = json_handler.fn 

    def preexec():
        try:
            # ensure the child dies if we are SIGKILLED
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        except:
            log("Failed to setup PR_SET_PDEATHSIG.")
            pass

    p = subprocess.Popen(args.pargs, preexec_fn=preexec, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True, env=env)
    CURRENT_PROC.append(p)

    zlog = None
    if args.zlog:
        try:
            relpath = os.path.dirname(__file__)
            sys.path.append(relpath)
            from ztee import GZipLog
            
            zlog_ = GZipLog(log_file=args.zlog)
            es.update_doc({ "stdlog_gzip": args.zlog })

            log("Open gzip log file: %s" % args.zlog)
            zlog = zlog_
        except Exception as e:
            log("Failed to setup zlog file: " + str(e))

    es.update_doc({ "pid": p.pid })
    es.update_doc({ "monitoring_pid": os.getpid() })
    es.update_doc({ "monitoring_socket": json_handler.fn })
    es.defaults()
    es.make_report()

    log_handler = FDOutputListener(fd=p.stdout.fileno(), es=es, zlog=zlog, close_socket=json_handler)
    log_handler.handle_line("-- starting process: %s --\n" % str(args.pargs))

    try:
        #manager.event_loop(timeout=5, exit_fd=p.stdout.fileno())
        asyncore.loop(timeout=5)
    except select.error as e:
        # we have this on ctrl+c
        # just terminate the child
        log("Select error (we will terminate): " + str(e))
        p.terminate()

    # at this point the program is dead
    r =  p.wait()
    log_handler.handle_line("\n-- process exit: %s --\n" % str(r))
    log_handler.finish()

    es.update_doc({ "exit_code": r })
    es.make_report()

    CURRENT_PROC.remove(p)
    return r

def handle_signal(signum, frame):
    for proc in CURRENT_PROC:
        proc.send_signal(signum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor a child process and produce es documents.")
    parser.add_argument('--debug', '-d', action='store_true', help="Debug mode")
    parser.add_argument('--zlog', '-z', type=str, default=None, help="Don't output anything, zip the log file (uses ztee.py).")
    parser.add_argument('--path', '-p', type=str, default="/tmp/dqm_monitoring/", help="Path for the monitoring output.")
    parser.add_argument('pargs', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.pargs:
        parser.print_help()
        sys.exit(-1)
    elif args.pargs[0] == "--":
        # compat with 2.6
        args.pargs = args.pargs[1:]

    # do some signal magic
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    sys.exit(launch_monitoring(args))
