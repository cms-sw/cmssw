#!/usr/bin/env python3

from __future__ import print_function
import os
import sys
import argparse
import asyncore
import pickle
import logging
import subprocess
import shutil
import re
import collections
import json
import tempfile
import signal
import time
import glob

# Utilities
log_format = '%(asctime)s: %(name)-20s - %(levelname)-8s - %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)
root_log = logging.getLogger()

class Applet(object):
    def __init__(self, name, opts, **kwargs):
        self.name = name
        self.opts = opts
        self.kwargs = kwargs

        self.do_init()

    def write(self, fp):
        self.control_fp = fp

        with open(fp, "wb") as f:
            pickle.dump(self, f)

        self.log.info("Written control file: %s", fp)

    @staticmethod
    def read(fp):
        with open(fp, "rb") as f:
            return pickle.load(f)

    @property
    def log(self):
        return logging.getLogger(self.name)

    def do_init(self):
        pass

    def do_exec(self):
        pass

def preexec_kill_on_pdeath():
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)

# Actual implementation of the workers

class Playback(Applet):
    re_pattern = re.compile(r'run([0-9]+)_ls([0-9]+)_stream([A-Za-z0-9]+)_([A-Za-z0-9_-]+)\.jsn')

    def discover_files(self):
        self.lumi_found = {}

        files_found = set()
        streams_found = set()
        run_found = None
        for f in os.listdir(self.input):
            r = self.re_pattern.match(f)
            if r:
                run, lumi, stream, stream_source = r.groups()
                run, lumi = int(run), int(lumi)

                if run_found is None:
                    run_found = run
                elif run_found != run:
                    raise Exception("Files from multiple runs are not (yet) supported for as playback input.")

                lumi_dct = self.lumi_found.setdefault(lumi, { 'streams': {} })
                lumi_dct["streams"][stream] = (f, stream_source)
                files_found.add(f)
                streams_found.add(stream)

        if run_found is None:
            raise Exception("Playback files not found.")

        if self.run < 0:
            self.run = run_found

        self.log.info("Found run %s, will map output to run %s", run_found, self.run)
        self.log.info("Found %d lumisections with %d files", len(self.lumi_found), len(files_found))
        self.log.info("Found %d streams: %s", len(streams_found), list(streams_found))

        self.lumi_order = list(self.lumi_found.keys())
        self.lumi_order.sort()
        self.log.info("Lumi order: %s", str(self.lumi_order))

    def do_init(self):
        # check if our input directory is okay
        self.input = self.opts.playback
        self.ramdisk = self.opts.work_ramdisk
        self.run = self.opts.run
        self.log.info("Using input directory: %s", self.input)

        self.discover_files()

        self.output = os.path.join(self.ramdisk, "run%06d" % self.run)
        if not os.path.isdir(self.output):
            os.makedirs(self.output)
        self.log.info("Using output directory: %s", self.output)

        self.global_file = os.path.join(self.ramdisk, ".run%06d.global" % self.run)
        self.log.info("Writing: %s", self.global_file)
        with open(self.global_file, "w") as f:
            f.write("run_key = pp_run")

        self.lumi_backlog = collections.deque()
        self.lumi_backlog_size = 10
        self.next_lumi_index = 1

    def do_create_lumi(self):
        orig_lumi = self.lumi_order[(self.next_lumi_index - 1) % len(self.lumi_order)]
        play_lumi = self.next_lumi_index;
        self.next_lumi_index += 1

        self.log.info("Start copying lumi (original) %06d -> %06d (playback)", orig_lumi, play_lumi)

        lumi_dct = self.lumi_found[orig_lumi]
        streams = lumi_dct["streams"]

        def ijoin(f):
            return os.path.join(self.input, f)

        def ojoin(f):
            return os.path.join(self.output, f)

        written_files = set()
        for stream, v  in streams.items():
            jsn_orig_fn, stream_source = v
            jsn_play_fn = "run%06d_ls%04d_stream%s_%s.jsn" % (self.run, play_lumi, stream, stream_source)

            # define dat filename
            ext = "dat"
            if stream.startswith("streamDQMHistograms"):
                ext = "pb"
            dat_play_fn = "run%06d_ls%04d_stream%s_%s.%s" % (self.run, play_lumi, stream, stream_source, ext)

            # read the original file name, for copying
            with open(ijoin(jsn_orig_fn), 'r') as f:
                jsn_data = json.load(f)
                dat_orig_fn = jsn_data["data"][3]

            # copy the data file
            if os.path.exists(ijoin(dat_orig_fn)):
                self.log.info("C: %s -> %s", dat_orig_fn, dat_play_fn)
                shutil.copyfile(ijoin(dat_orig_fn), ojoin(dat_play_fn))

                written_files.add(dat_play_fn)
            else:
                log.warning("Dat file is missing: %s", dat_orig_fn)

            # write a new json file point to a different data file
            # this has to be atomic!
            jsn_data["data"][3] = dat_play_fn

            f = tempfile.NamedTemporaryFile(prefix=jsn_play_fn+ ".", suffix=".tmp", dir = self.output, delete=False)
            tmp_fp = f.name
            json.dump(jsn_data, f)
            f.close()

            os.rename(tmp_fp, ojoin(jsn_play_fn))
            written_files.add(jsn_play_fn)

        self.log.info("Copied %d files for lumi %06d", len(written_files), play_lumi)

        self.lumi_backlog.append((play_lumi, written_files))
        while len(self.lumi_backlog) > self.lumi_backlog_size:
            old_lumi, files_to_delete = self.lumi_backlog.popleft()

            self.log.info("Deleting %d files for old lumi %06d", len(files_to_delete), old_lumi)
            for f in files_to_delete:
                os.unlink(ojoin(f))

    def do_exec(self):
        last_write = 0
        lumi_produced = 0

        while True:
            time.sleep(1)

            now = time.time()
            if (now - last_write) > self.opts.playback_time_lumi:
                last_write = now

                if self.opts.playback_nlumi > -1 and lumi_produced >= self.opts.playback_nlumi:
                    break

                self.do_create_lumi()
                lumi_produced += 1

        # write eor
        eor_fn = "run%06d_ls0000_EoR.jsn" % (self.run, )
        eor_fp = os.path.join(self.output, eor_fn)
        with open(eor_fp, "w"):
            pass

        self.log.info("Wrote EoR: %s", eor_fp)

start_dqm_job = """
#!/bin/env /bin/bash
set -x #echo on
TODAY=$(date)
logname="/var/log/hltd/pid/hlt_run$4_pid$$.log"
lognamez="/var/log/hltd/pid/hlt_run$4_pid$$_gzip.log.gz"
#override the noclobber option by using >| operator for redirection - then keep appending to log
echo startDqmRun invoked $TODAY with arguments $1 $2 $3 $4 $5 $6 $7 $8 >| $logname
export http_proxy="http://cmsproxy.cms:3128"
export https_proxy="https://cmsproxy.cms:3128/"
export NO_PROXY=".cms"
export SCRAM_ARCH=$2
cd $1
cd base
source cmsset_default.sh >> $logname
cd $1
cd current
pwd >> $logname 2>&1
eval `scram runtime -sh`;
cd $3;
pwd >> $logname 2>&1
#exec esMonitoring.py -z $lognamez cmsRun `readlink $6` runInputDir=$5 runNumber=$4 $7 $8 >> $logname 2>&1
exec esMonitoring.py cmsRun `readlink $6` runInputDir=$5 runNumber=$4 $7 $8
"""

start_dqm_job = start_dqm_job.replace("/var/log/hltd/pid/", '{log_path}/')
start_dqm_job = start_dqm_job.replace(" cmsRun ", ' {cmsRun} ')


RunDesc = collections.namedtuple('Run', ['run', 'run_fp', 'global_fp', 'global_param'])
RunState = collections.namedtuple('RunState', ['desc', 'proc'])

class FrameworkJob(Applet):
    def _set_name(self):
        x = os.path.basename(self.cfg_file)
        x = re.sub(r'(.*)\.py', r'\1', x)
        x = re.sub(r'(.*)_cfg', r'\1', x)
        x = re.sub(r'(.*)-live', r'\1', x)
        x = re.sub(r'(.*)_sourceclient', r'\1', x)
        x = re.sub(r'(.*)_dqm', r'\1', x)

        x = "".join([c for c in x if c.isalnum()])
        self.tag = x
        self.name = "cmssw_%s" % x

    def _find_release(self):
        fp = os.path.realpath(self.cfg_file)
        while len(fp) > 3:
            bn = os.path.basename(fp)
            fp = os.path.dirname(fp)

            if bn == "src":
                break

        if len(fp) <= 3:
            raise Exception("Could not find the cmssw release area.")

        self.cmsenv_path = fp
        self.log.info("cmsenv path: %s", self.cmsenv_path)

    def _prepare_files(self):
        self.home_path = os.path.join(self.opts.work_home, "%s_%s" % (self.name, hex(id(self))))
        self.home_path = os.path.realpath(self.home_path)
        os.makedirs(self.home_path)

        self.log_path = self.opts.work_logs
        self.log.info("logs path: %s", self.log_path)

        self.exec_file = os.path.join(self.home_path, "startDqmRun.sh")
        self.log.info("Creating: %s", self.exec_file)
        f = open(self.exec_file, "w")
        template = start_dqm_job
        body = template.format(log_path=self.log_path, cmsRun=self.opts.cmsRun)
        f.write(body)
        f.close()
        os.chmod(self.exec_file, 0o755)

        cmsset_globs = ["/afs/cern.ch/cms/cmsset_default.sh", "/home/dqm*local/base/cmsset_default.sh"]
        cmsset_target = None
        for t in cmsset_globs:
            files =  glob.glob(t)
            for f in files:
                cmsset_target = f
                break

        if cmsset_target is not None:
            base = os.path.join(self.home_path, "base")
            os.makedirs(base)

            cmsset_link = os.path.join(base, "cmsset_default.sh")
            self.log.info("Linking : %s -> %s", cmsset_link, cmsset_target)
            os.symlink(cmsset_target, cmsset_link)
        else:
            self.log.warning("Couldn't find cmsset_default.sh, source it yourself!")

        current_link = os.path.join(self.home_path, "current")
        target = os.path.relpath(self.cmsenv_path, self.home_path)
        self.log.info("Linking : %s -> %s", current_link, target)
        os.symlink(target, current_link)

        # check if current is outside the release directory
        # otherwise scram gets stuck forever
        cp = os.path.commonprefix([self.home_path, self.cmsenv_path])
        if self.cmsenv_path == cp:
            self.log.error("Working directory (incl. control directory), have to be outside the cmssw release. Otherwise scram fails due to recursive links.")
            raise Exception("Invalid home_path: %s" % self.home_path)

        output_link = os.path.join(self.home_path, "output")
        output_target = os.path.realpath(self.opts.work_output)
        target = os.path.relpath(output_target, self.home_path)
        self.log.info("Linking : %s -> %s", output_link, target)
        os.symlink(target, output_link)
        self.output_path = output_link

        cfg_link = os.path.join(self.home_path, os.path.basename(self.cfg_file))
        target = self.cfg_fp
        self.log.info("Linking : %s -> %s", cfg_link, target)
        os.symlink(target, cfg_link)
        self.cfg_link = cfg_link


    def do_init(self):
        # check if our input directory is okay
        self.ramdisk = self.opts.work_ramdisk
        self.run = self.opts.run
        self.cfg_file = self.kwargs["cfg_file"]

        if not os.path.isfile(self.cfg_file):
            raise Exception("Configuration file not found: %s" % self.cfg_file)

        self.cfg_fp = os.path.realpath(self.cfg_file)
        self.ramdisk_fp = os.path.realpath(self.ramdisk)

        self._set_name()
        self._find_release()
        self._prepare_files()

    def make_args(self, run):
        args = []
        args.append("bash")                 # arg 0
        args.append(self.exec_file)         # arg 0
        args.append(self.home_path)         # home path
        args.append("slc6_amd64_gcc491")    # release
        args.append(self.output_path)       # cwd/output path
        args.append(str(run))               # run
        args.append(self.ramdisk_fp)        # ramdisk
        args.append(self.cfg_link)          # cmsRun arg 1
        args.append("runkey=pp_run")        # cmsRun arg 2

        return args

    def discover_latest(self):
        re_run = re.compile(r'run([0-9]+)')
        re_global = re.compile(r'\.run([0-9]+)\.global')

        # find runs
        runs = {}
        globals = {}
        for x in os.listdir(self.ramdisk):
            m = re_run.match(x)
            if m:
                runs[int(m.group(1))] = x

            m = re_global.match(x)
            if m:
                globals[int(m.group(1))] = x

        # find max global for which there is a run directory
        run_set = set(runs.keys())
        run_set = run_set.intersection(globals.keys())

        if self.opts.run < 0:
            largest = max(run_set)
        else:
            largest = self.opts.run

        #self.log.info("Largest: %s", largest)
        global_fp = os.path.join(self.ramdisk, globals[largest])
        with open(global_fp, "r") as f:
            global_param = f.read()

        return RunDesc(
            run = largest,
            run_fp = os.path.join(self.ramdisk, runs[largest]),
            global_fp = global_fp,
            global_param = global_param,
        )

    def start_run(self, current):
        old_state = self.current_state

        # kill the old run
        # nope, since it involves eof and i am lazy
        if old_state:
            return

        args = self.make_args(current.run)
        self.log.info("Executing: %s", " ".join(args))
        proc = subprocess.Popen(args, preexec_fn=preexec_kill_on_pdeath)
        self.current_state = RunState(desc=current, proc=proc)

    def do_exec(self):
        time.sleep(1)

        self.current_state = None

        while True:
            latest = self.discover_latest()
            if self.current_state is None or latest != self.current_state.desc:
                self.log.info("Found latest run: %s", latest)

                self.start_run(latest)

            if not self.current_state:
                self.log.info("Run not found, waiting 1 sec.")
            else:
                r = self.current_state.proc.poll()
                if r is not None:
                    self.log.info("Process exitted: %s", r)

                    return 0

            time.sleep(1)

import getpass
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[-1].endswith(".pkl"):
        f = sys.argv[-1]
        obj = Applet.read(f)

        ret = obj.do_exec()
        sys.exit(ret if ret else 0)

    # control -> interal files and home directory for the run
    subdirectories = ["ramdisk", "output", "control", "home", "logs", "dqm_monitoring"]
    username = getpass.getuser()

    parser = argparse.ArgumentParser(description="Emulate DQM@P5 environment and launch cmssw jobs.")
    #parser.add_argument('-q', action='store_true', help="Don't write to stdout, just the log file.")
    #parser.add_argument("log", type=str, help="Filename to write.", metavar="<logfile.gz>")

    parser.add_argument("--work", "-w", type=str, help="Working directory (used for inputs,outputs,monitoring and logs).", default="/tmp/pplay." + username)
    parser.add_argument("--clean", "-c", action="store_true", help="Clean work directories (if they are not set).", default=False)
    parser.add_argument("--dry", "-n", action="store_true", help="Do not execute, just init.", default=False)

    work_group = parser.add_argument_group('Paths', 'Path options for cmssw jobs, auto generated if not specified.')
    for subdirectory in subdirectories:
        work_group.add_argument("--work_%s" % subdirectory, type=str, help="Path for %s directory." % subdirectory, default=None)

    playback_group = parser.add_argument_group('Playback', 'Playback configuration/parameters.')
    playback_group.add_argument("--playback", "-p", type=str, metavar="PLAYBACK_INPUT_DIR", help="Enable playback (emulate file delivery, otherwise set work_input).", default=None)
    playback_group.add_argument("--playback_nlumi", type=int, help="Number of lumis to deliver, -1 for forever.", default=-1)
    playback_group.add_argument("--playback_time_lumi", type=float, help="Number of seconds between lumisections.", default=23.3)

    run_group = parser.add_argument_group('Run', 'Run configuration/parameters.')
    run_group.add_argument("--run", type=int, help="Run number, -1 for autodiscovery.", default=-1)
    run_group.add_argument("--cmsRun", type=str, help="cmsRun command to run, for igprof and so on.", default="cmsRun")

    parser.add_argument('cmssw_configs', metavar='cmssw_cfg.py', type=str, nargs='*', help='List of cmssw jobs (clients).')

    args = parser.parse_args()

    if len(args.cmssw_configs) and args.cmssw_configs[0] == "--":
        # compat with 2.6
        args.cmssw_configs = args.cmssw_configs[1:]

    for subdirectory in subdirectories:
        if getattr(args, "work_" + subdirectory) is None:
            setattr(args, "work_" + subdirectory, os.path.join(args.work, subdirectory))

            path = getattr(args, "work_" + subdirectory)
            if args.clean and os.path.isdir(path):
                root_log.info("Removing directory: %s", path)
                shutil.rmtree(path)

        path = getattr(args, "work_" + subdirectory)
        if not os.path.isdir(path):
            os.makedirs(path)

        root_log.info("Using directory: %s", path)

    print("*"*80)
    print(args)
    print("*"*80)

    applets = []

    if args.playback:
        # launch playback service
        playback = Playback("playback_emu", opts=args)
        applets.append(playback)

    for cfg in args.cmssw_configs:
        cfg_a = FrameworkJob("framework_job", opts=args, cfg_file=cfg)
        applets.append(cfg_a)

    if len(applets) == 0:
        sys.stderr.write("At least one process should be specified, use --playback and/or cmssw_configs options.\n")

    # serialize them into control directory
    for a in applets:
        fn = "%s_%s.pkl" % (a.name, hex(id(a)))
        a.write(os.path.join(args.work_control, fn))

    if args.dry:
        sys.exit(0)

    # launch each in a separate subprocess
    for a in applets:
        fp = a.control_fp

        args = [os.path.realpath(__file__), fp]
        a.control_proc = subprocess.Popen(args, preexec_fn=preexec_kill_on_pdeath)

    for a in applets:
        # wait till everything finishes
        a.control_proc.wait()

