import shutil
import os
import atexit
import signal
import time
import multiprocessing
import subprocess
import inspect

my_path = os.path.dirname(__file__)
cmsRunConf = os.path.join(my_path, "DQMStreamReader_replay_cfg.py")
source = {
    "run": 100,
    "root": "/build1/micius/OnlineDQM_sample/",
    "lumi": 3
}

def atomic_cp(src, dest):
    bp = os.path.dirname(dest)
    nm = os.path.basename(dest)

    nm += ".temp"
    tmppath = os.path.join(bp, nm)

    shutil.copyfile(src, tmppath)
    shutil.move(tmppath, dest)

class CMSRun(multiprocessing.Process):
    def __init__(self, args):
        multiprocessing.Process.__init__(self)
        self.queue = multiprocessing.Queue()
        self.args = args
        self.p = None

    def run(self):
        try:
            p = subprocess.Popen(self.args, stderr=subprocess.PIPE, bufsize=0)
            self.p = p

            for line in iter(p.stderr.readline, b''):
                line = line.strip()
                print "received stderr:", line
                self.queue.put(line)

            print "cmsRun exit, code: ", self.p.wait()

        except KeyboardInterrupt:
            if self.p and self.p.poll() is None:
                self.p.send_signal(signal.SIGTERM)

    def wait_for_line(self, lambdaf):
        print "**** waiting for the line:", lambdaf.func_code.co_consts, "****"
        rejected = []
        while True:
            line = self.queue.get()
            if lambdaf(line):
                return rejected
            else:
                rejected.append(line)

def spawn_cmsrun(**kwargs):
    args = [ "cmsRun", cmsRunConf ]
    for k, v in kwargs.items():
        args.append("%s=%s" % (k, v))

    print "args:", args

    t = CMSRun(args)
    t.start()

    return t 

def prepare_scenario():
    # use the caller name as a name 
    name = inspect.stack()[1][3]

    tmpDir = "/tmp/dqmtmp_%s" % name
    runDir = tmpDir + "/run%06d" % source["run"]
    shutil.rmtree(tmpDir, ignore_errors=True)

    os.mkdir(tmpDir)
    os.mkdir(runDir)

    in_prefix = source["root"] + "run%06d/run%06d_" % (source["run"], source["run"])
    out_prefix = runDir + "/run%06d_" % source["run"]

    print "input:", in_prefix
    print "output:", out_prefix

    return tmpDir, runDir, in_prefix, out_prefix

def scenario1():
    tmpDir, runDir, in_prefix, out_prefix = prepare_scenario()
    p = spawn_cmsrun(runNumber=source["run"], runInputDir=tmpDir)

    p.wait_for_line(lambda x: x.startswith("Checking eor file"))

    # copy the first lumi
    atomic_cp(in_prefix + "ls0001_streamA.dat", out_prefix + "ls0001_streamA.dat")
    atomic_cp(in_prefix + "ls0001.jsn", out_prefix + "ls0001.jsn")

    p.wait_for_line(lambda x: "Run 100, Event 4, LumiSection 1" in x)
    p.wait_for_line(lambda x: "Run 100, Event 14, LumiSection 1" in x)
    p.wait_for_line(lambda x: "Run 100, Event 94, LumiSection 1" in x)

    # copy the third lumi
    atomic_cp(in_prefix + "ls0003_streamA.dat", out_prefix + "ls0003_streamA.dat")
    atomic_cp(in_prefix + "ls0003.jsn", out_prefix + "ls0003.jsn")

    p.wait_for_line(lambda x: "waiting for the next LS" in x)

    # copy the second lumi
    atomic_cp(in_prefix + "ls0002_streamA.dat", out_prefix + "ls0002_streamA.dat")
    atomic_cp(in_prefix + "ls0002.jsn", out_prefix + "ls0002.jsn")
    p.wait_for_line(lambda x: "Run 100, Event 104, LumiSection 2" in x)

    # we don't have a 3 lumi file, this is just first dat file
    rejects = p.wait_for_line(lambda x: "Run 100, Event 4, LumiSection 1" in x)

    # only one events from lumi 2 should be processed
    assert not ("LumiSection 2" in "".join(rejects))

    # copy the end of run and exit
    atomic_cp(in_prefix + "ls0000_EoR.jsn", out_prefix + "ls0000_EoR.jsn")
    p.wait_for_line(lambda x: "Streamer state changed: 0 -> 2" in x)

def scenario2_end_of_run_kills():
    tmpDir, runDir, in_prefix, out_prefix = prepare_scenario()
    p = spawn_cmsrun(runNumber=source["run"], runInputDir=tmpDir, endOfRunKills="True")

    p.wait_for_line(lambda x: x.startswith("Checking eor file"))

    # copy the first lumi
    atomic_cp(in_prefix + "ls0001_streamA.dat", out_prefix + "ls0001_streamA.dat")
    atomic_cp(in_prefix + "ls0001.jsn", out_prefix + "ls0001.jsn")


    p.wait_for_line(lambda x: "Run 100, Event 4, LumiSection 1" in x)
    p.wait_for_line(lambda x: "Run 100, Event 14, LumiSection 1" in x)
    p.wait_for_line(lambda x: "Run 100, Event 94, LumiSection 1" in x)

    # copy the end of run 
    atomic_cp(in_prefix + "ls0000_EoR.jsn", out_prefix + "ls0000_EoR.jsn")

    # copy the second lumi
    atomic_cp(in_prefix + "ls0002_streamA.dat", out_prefix + "ls0002_streamA.dat")
    atomic_cp(in_prefix + "ls0002.jsn", out_prefix + "ls0002.jsn")
 
    rejected = p.wait_for_line(lambda x: "Streamer state changed: 0 -> 1" in x)

    # second lumi should not be processed 
    assert not ("LumiSection 2" in "".join(rejected))

def scenario3_no_data_file():
    tmpDir, runDir, in_prefix, out_prefix = prepare_scenario()
    p = spawn_cmsrun(runNumber=source["run"], runInputDir=tmpDir, endOfRunKills="True")

    p.wait_for_line(lambda x: x.startswith("Checking eor file"))

    # copy the first lumi
    atomic_cp(in_prefix + "ls0001.jsn", out_prefix + "ls0001.jsn")

    # copy the second lumi
    atomic_cp(in_prefix + "ls0002_streamA.dat", out_prefix + "ls0002_streamA.dat")
    atomic_cp(in_prefix + "ls0002.jsn", out_prefix + "ls0002.jsn")
    
    p.wait_for_line(lambda x: "Run 100, Event 104, LumiSection 2" in x)
 
    # copy the end of run 
    atomic_cp(in_prefix + "ls0000_EoR.jsn", out_prefix + "ls0000_EoR.jsn")
    p.wait_for_line(lambda x: "Streamer state changed: 0 -> 1" in x)

def scenario4_end_of_run_in_constructor():
    tmpDir, runDir, in_prefix, out_prefix = prepare_scenario()
    p = spawn_cmsrun(runNumber=source["run"], runInputDir=tmpDir, endOfRunKills="True")

    p.wait_for_line(lambda x: x.startswith("Checking eor file"))
    atomic_cp(in_prefix + "ls0000_EoR.jsn", out_prefix + "ls0000_EoR.jsn")

    # copy the end of run 
    p.wait_for_line(lambda x: "Streamer state changed: 0 -> 1" in x)


tests = [
    scenario1,
    scenario2_end_of_run_kills,
    scenario3_no_data_file,
    scenario4_end_of_run_in_constructor,
]

if __name__ == "__main__":
    for test in tests:
        test()

#scenario1()
#scenario2_end_of_run_kills()
#scenario3_no_data_file()
#scenario4_end_of_run_in_constructor()
