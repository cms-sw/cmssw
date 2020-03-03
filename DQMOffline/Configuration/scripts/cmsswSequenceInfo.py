#!/usr/bin/env python3
import re
import shutil
import sqlite3
import tempfile
import functools
import subprocess
from multiprocessing.pool import ThreadPool

tp = ThreadPool()
stp = ThreadPool()

# This file will actually be opened, though the content does not matter. Only to make CMSSW start up at all.
INFILE = "/store/data/Run2018A/EGamma/RAW/v1/000/315/489/00000/004D960A-EA4C-E811-A908-FA163ED1F481.root"

# Modules that will be loaded but do not come from the DQM Sequence.
BLACKLIST='^(TriggerResults|.*_step|DQMoutput|siPixelDigis)$'

# HARVESTING does not work for now, since the jobs crash at the end. this should be easy to work around, somehow.
RELEVANTSTEPS = ["DQM", "VALIDATION"]

@functools.lru_cache(maxsize=None)
def inspectsequence(sequence, step, era, scenario):
  sep = ":"
  if not sequence:
    sep = ""
  type = ""
  if step == "DQM":
    type = "--data"
  otherstep = ""
  if step != "HARVESTING":
    otherstep = "RAW2DIGI:siPixelDigis,"

  wd = tempfile.mkdtemp()


  # run cmsdriver
  driverargs = [
    "cmsDriver.py",
    "step3",
    "--conditions",  "auto:run2_data",                                         # conditions is mandatory, but should not affect the result.
    "-s", otherstep+step+sep+sequence,                                         # running only DQM seems to be not possible, so also load a single module for RAW2DIGI
    "--process", "DUMMY", type, "--era" if era else "", era,                   # random switches, era is important as it trigger e.g. switching phase0/pahse1/phase2
    "--eventcontent", "DQM",  "--scenario" if scenario else "", scenario,  "--datatier",  "DQMIO", # more random switches, sceanario should affect which DQMOffline_*_cff.py is loaded
    "--customise_commands", 'process.Tracer = cms.Service("Tracer")',          # the tracer will tell us which modules actually run
    "--runUnscheduled",                                                        # convert everything to tasks. Used in production, which means sequence ordering does not matter!
    "--filein", INFILE, "-n", "0",                                             # load an input file, but do not process any events -- it would fail anyways.
    "--python_filename", "cmssw_cfg.py", "--no_exec"
  ]
  # filter out empty args
  driverargs = [x for x in driverargs if x]
  subprocess.check_call(driverargs, cwd=wd)

  # run cmsRun to get module list
  tracedump = subprocess.check_output(["cmsRun", "cmssw_cfg.py"], stderr=subprocess.STDOUT, cwd=wd)
  lines = tracedump.splitlines()
  labelre = re.compile(b"[+]+ starting: constructing module with label '(\w+)'")
  blacklistre = re.compile(BLACKLIST)
  modules = []
  for line in lines:
    m = labelre.match(line)
    if m:
      label = m.group(1).decode()
      if blacklistre.match(label):
        continue
      modules.append(label)

  modules = set(modules)

  # run edmConfigDump to get module config
  configdump = subprocess.check_output(["edmConfigDump", "cmssw_cfg.py"], cwd=wd)
  lines = configdump.splitlines()
  modulere = re.compile(b'process[.](.*) = cms.ED.*\("(.*)",')

  # collect the config blocks out of the config dump.
  modclass = dict()
  modconfig = dict()
  inconfig = None
  for line in lines:
    if inconfig:
      modconfig[inconfig] += line
      if line == b')':
        inconfig = None
      continue

    m = modulere.match(line)
    if m:
      label = m.group(1).decode()
      plugin = m.group(2).decode()
      if label in modules:
        modclass[label] = plugin
        modconfig[label] = line
        inconfig = label

  # run edmPluginHelp to get module properties
  plugininfo = tp.map(getplugininfo, modclass.values())

  shutil.rmtree(wd)

  return modconfig, modclass, dict(plugininfo)

@functools.lru_cache(maxsize=None)
def getplugininfo(pluginname):
  plugindump = subprocess.check_output(["edmPluginHelp", "-p", pluginname])
  line = plugindump.splitlines()[0].decode()
  pluginre = re.compile(".* " + pluginname + ".*[(](\w+)::(\w+)[)]")
  m = pluginre.match(line)
  if not m:
    return (pluginname, "", "")
  else:
    return (pluginname, (m.group(1), m.group(2)))

def formatsequenceinfo(modconfig, modclass, plugininfo):
  for label in sorted(modclass.keys()):
    print("Module %s of class %s family %s base %s config %s" % (label, modclass[label], plugininfo[modclass[label]][0], plugininfo[modclass[label]][1], modconfig[label]))

DBSCHEMA = """
  CREATE TABLE IF NOT EXISTS plugin(classname, edmfamily, edmbase);
  CREATE UNIQUE INDEX IF NOT EXISTS plugins ON plugin(classname);
  CREATE TABLE IF NOT EXISTS module(id INTEGER PRIMARY KEY, classname, instancename, variation, config);
  CREATE UNIQUE INDEX IF NOT EXISTS modules ON module(instancename, variation); 
  CREATE UNIQUE INDEX IF NOT EXISTS configs ON module(config); 
  CREATE TABLE IF NOT EXISTS sequence(id INTEGER PRIMARY KEY, name, step, era, scenario);
  CREATE UNIQUE INDEX IF NOT EXISTS squences ON sequence(name, step, era, scenario);
  CREATE TABLE IF NOT EXISTS workflow(wfid, sequenceid);
  CREATE TABLE IF NOT EXISTS sequencemodule(moduleid, sequenceid);
"""

def storesequenceinfo(seqname, step, era, scenario, modconfig, modclass, plugininfo):
  with sqlite3.connect("sequences.db") as db:
    cur = db.cursor()
    cur.executescript(DBSCHEMA)
    # first, check if we already have that one. Ideally we'd check before doing all the work, but then the lru cache will take care of that on a different level.
    seqid = list(cur.execute("SELECT id FROM sequence WHERE (name, step, era, scenario) = (?, ?, ?, ?);", (seqname, step, era, scenario)))
    if seqid:
      return

    cur.execute("BEGIN;")
    cur.execute("CREATE TEMP TABLE newmodules(instancename, classname, config);")
    cur.executemany("INSERT INTO newmodules VALUES (?, ?, ?)", ((label, modclass[label], modconfig[label]) for label in modconfig))
    cur.execute("""
      INSERT OR IGNORE INTO module(classname, instancename, variation, config) 
      SELECT classname, instancename, 
        (SELECT count(*) FROM module AS existing WHERE existing.instancename = newmodules.instancename), 
        config FROM newmodules;
    """)

    cur.executemany("INSERT OR IGNORE INTO plugin VALUES (?, ?, ?);", ((plugin, edm[0], edm[1]) for plugin, edm in plugininfo.items()))
    cur.execute("INSERT OR FAIL INTO sequence(name, step, era, scenario) VALUES(?, ?, ?, ?);", (seqname, step, era, scenario))
    seqid = list(cur.execute("SELECT id FROM sequence WHERE (name, step, era, scenario) = (?, ?, ?, ?);", (seqname, step, era, scenario)))
    seqid = seqid[0][0]
    cur.executemany("INSERT INTO sequencemodule SELECT id, ? FROM module WHERE config = ?;", ((seqid, modconfig[label]) for label in modconfig))
    cur.execute("COMMIT;")

def inspectworkflows(wfnumber):
  # dump steps
  sequences = []
  if wfnumber:
    stepdump = subprocess.check_output(["runTheMatrix.py", "-l", str(wfnumber), "-ne"])
  else:
    stepdump = subprocess.check_output(["runTheMatrix.py", "-ne"])
  lines = stepdump.splitlines()
  for line in lines:
    if not b'cmsDriver.py' in line: continue
    args = list(reversed(line.decode().split(" ")))
    step = ""
    scenario = ""
    era = ""
    while args:
      item = args.pop()
      if item == '-s':
        step = args.pop()
      if item == '--scenario':
        scenario = args.pop()
      if item == '--era':
        era = args.pop()
    steps = step.split(",")
    for step in steps:
      s = step.split(":")[0]
      if s in RELEVANTSTEPS:
        if ":" in step:
          seqs = step.split(":")[1]
          for seq in seqs.split("+"):
            sequences.append((seq, s, era, scenario))
        else:
          sequences.append(("", s, era, scenario))
  return sequences

def processseqs(seqs):
  # launch one map_async per element to get finer grain tasks
  tasks = [stp.map_async(lambda seq: (seq, inspectsequence(*seq)), [seq]) for seq in seqs]

  while tasks:
    running = []
    done = []
    for t in tasks:
      if t.ready():
        done.append(t)
      else:
        running.append(t)
    for t in done:
      if not t.successful():
        print("Task failed.")
      for it in t.get(): # should only be one
        seq, res = it
        storesequenceinfo(*seq, *res)
    tasks = running


def serve():
  pass


if __name__ == "__main__":

  import argparse
  parser = argparse.ArgumentParser(description='Collect information about DQM sequences.')
  parser.add_argument("--sequence", default="", help="Name of the sequence")
  parser.add_argument("--step", default="DQM", help="cmsDriver step that the sequence applies to")
  parser.add_argument("--era", default="Run2_2018", help="CMSSW Era to use")
  parser.add_argument("--scenario", default="pp", help="cmsCriver scenario")
  parser.add_argument("--workflow", default=None, help="Ignore other options and inspect this workflow instead.")
  parser.add_argument("--runTheMatrix", default=False, action="store_true", help="Ignore other options and inspect the full matrix instead.")

  args = parser.parse_args()

  if args.workflow:
    seqs = inspectworkflows(args.workflow)
    print("Analyzing %d seqs..." % len(seqs))
    processseqs(seqs)

  elif args.runTheMatrix:
    seqs = set(inspectworkflows(None))
    print("Analyzing %d seqs..." % len(seqs))
    processseqs(seqs)

  else:
    modconfig, modclass, plugininfo = inspectsequence(args.sequence, args.step, args.era, args.scenario)
    #formatsequenceinfo(modconfig, modclass, plugininfo)

    #modconfig = {"AAna": b"blib", "BAna": b"blaub"}
    #modclass = {"AAna": "DQMA", "BAna": "DQMB"}
    #plugininfo = {"DQMA": ("legacy", "EDAnalyzer"), "DQMB": ("one", "EDProducer")}
    storesequenceinfo(args.sequence, args.step, args.era, args.scenario, modconfig, modclass, plugininfo), 

