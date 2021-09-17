#!/usr/bin/env python3
import os
import re
import time
import shutil
import sqlite3
import tempfile
import functools
import subprocess
from collections import namedtuple
from collections import defaultdict
from multiprocessing.pool import ThreadPool

Sequence = namedtuple("Sequence", ["seqname", "step", "era", "scenario", "mc", "data", "fast"])

# We use two global thread pools, to avoid submitting from one Pool into itself.
tp = ThreadPool()
stp = ThreadPool()

# SQLiteDB to write results to.
# Set later from commandline args.
DBFILE = None 

# This file will actually be opened, though the content does not matter. Only to make CMSSW start up at all.
INFILE = "/store/data/Run2018A/EGamma/RAW/v1/000/315/489/00000/004D960A-EA4C-E811-A908-FA163ED1F481.root"

# Modules that will be loaded but do not come from the DQM Sequence.
BLACKLIST='^(TriggerResults|.*_step|DQMoutput|siPixelDigis)$'

# Set later from commandline args
RELEVANTSTEPS = []

@functools.lru_cache(maxsize=None)
def inspectsequence(seq):
    sep = ":"
    if not seq.seqname:
        sep = ""

    wd = tempfile.mkdtemp()

    # Provide a fake GDB to prevent it from running if cmsRun crashes. It would not hurt to have it run but it takes forever.
    with open(wd + "/gdb", "w"):
        pass
    os.chmod(wd + "/gdb", 0o700)
    env = os.environ.copy()
    env["PATH"] = wd + ":" + env["PATH"]

    # run cmsdriver
    driverargs = [
        "cmsDriver.py",
        "step3",
        "--conditions", "auto:run2_data",                                    # conditions is mandatory, but should not affect the result.
        "-s", seq.step+sep+seq.seqname,                            # running only DQM seems to be not possible, so also load a single module for RAW2DIGI
        "--process", "DUMMY", 
        "--mc" if seq.mc else "", "--data" if seq.data else "", "--fast" if seq.fast else "", # random switches 
        "--era" if seq.era else "", seq.era,                                 # era is important as it trigger e.g. switching phase0/pahse1/phase2
        "--eventcontent", "DQM", "--scenario" if seq.scenario else "", seq.scenario, # sceanario should affect which DQMOffline_*_cff.py is loaded
        "--datatier", "DQMIO",                                               # more random switches, 
        "--customise_commands", 'process.Tracer = cms.Service("Tracer")',    # the tracer will tell us which modules actually run
        "--filein", INFILE, "-n", "0",                                       # load an input file, but do not process any events -- it would fail anyways.
        "--python_filename", "cmssw_cfg.py", "--no_exec"
    ]
    # filter out empty args
    driverargs = [x for x in driverargs if x]
    subprocess.check_call(driverargs, cwd=wd, stdout=2) # 2: STDERR

    # run cmsRun to get module list
    proc = subprocess.Popen(["cmsRun", "cmssw_cfg.py"], stderr=subprocess.STDOUT, stdout=subprocess.PIPE, cwd=wd, env=env)
    tracedump, _ = proc.communicate()
    # for HARVESTING, the code in endJob makes most jobs crash. But that is fine,
    # we have the data we need by then.
    if proc.returncode and seq.step not in ("HARVESTING", "ALCAHARVEST"):
        raise Exception("cmsRun failed for cmsDriver command %s" % driverargs)

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
            modconfig[inconfig] += b'\n' + line
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

    # clean up the temp dir in the end.
    shutil.rmtree(wd)

    return modconfig, modclass, dict(plugininfo)

# using a cache here to avoid running the (rather slow) edmPluginHelp multiple
# times for the same module (e.g. across different wf).
@functools.lru_cache(maxsize=None)
def getplugininfo(pluginname):
    plugindump = subprocess.check_output(["edmPluginHelp", "-p", pluginname])
    line = plugindump.splitlines()[0].decode()
    # we care only about the edm base class for now.
    pluginre = re.compile(".* " + pluginname + ".*[(]((\w+)::)?(\w+)[)]")
    m = pluginre.match(line)
    if not m:
        # this should never happen, but sometimes the Tracer does report things that are not actually plugins. 
        return (pluginname, ("", ""))
    else:
        return (pluginname, (m.group(2), m.group(3)))

def formatsequenceinfo(modconfig, modclass, plugininfo, showlabel, showclass, showtype, showconfig):
    # printing for command-line use.
    out = []
    for label in modclass.keys():
        row = []
        if showlabel:
            row.append(label)
        if showclass:
            row.append(modclass[label])
        if showtype:
            row.append("::".join(plugininfo[modclass[label]]))
        if showconfig:
            row.append(modconfig[label].decode())
        out.append(tuple(row))
    for row in sorted(set(out)):
        print("\t".join(row))

# DB schema for the HTML based browser. The Sequence members are kept variable
# to make adding new fields easy.
SEQFIELDS = ",".join(Sequence._fields)
SEQPLACEHOLDER = ",".join(["?" for f in Sequence._fields]) 
DBSCHEMA = f"""
    CREATE TABLE IF NOT EXISTS plugin(classname, edmfamily, edmbase);
    CREATE UNIQUE INDEX IF NOT EXISTS plugins ON plugin(classname);
    CREATE TABLE IF NOT EXISTS module(id INTEGER PRIMARY KEY, classname, instancename, variation, config);
    CREATE UNIQUE INDEX IF NOT EXISTS modules ON module(instancename, variation); 
    CREATE UNIQUE INDEX IF NOT EXISTS configs ON module(config); 
    CREATE TABLE IF NOT EXISTS sequence(id INTEGER PRIMARY KEY, {SEQFIELDS});
    CREATE UNIQUE INDEX IF NOT EXISTS squences ON sequence({SEQFIELDS});
    CREATE TABLE IF NOT EXISTS workflow(wfid, sequenceid);
    CREATE UNIQUE INDEX IF NOT EXISTS wrokflows ON workflow(sequenceid, wfid);
    CREATE TABLE IF NOT EXISTS sequencemodule(moduleid, sequenceid);
"""

def storesequenceinfo(seq, modconfig, modclass, plugininfo):
    with sqlite3.connect(DBFILE) as db:
        cur = db.cursor()
        cur.executescript(DBSCHEMA)
        # first, check if we already have that one. Ideally we'd check before doing all the work, but then the lru cache will take care of that on a different level.
        seqid = list(cur.execute(f"SELECT id FROM sequence WHERE ({SEQFIELDS}) = ({SEQPLACEHOLDER});", (seq)))
        if seqid:
            return

        cur.execute("BEGIN;")
        # dump everything into a temp table first... 
        cur.execute("CREATE TEMP TABLE newmodules(instancename, classname, config);")
        cur.executemany("INSERT INTO newmodules VALUES (?, ?, ?)", ((label, modclass[label], modconfig[label]) for label in modconfig))
        # ... then deduplicate and version the configs in plain SQL. 
        cur.execute("""
            INSERT OR IGNORE INTO module(classname, instancename, variation, config) 
            SELECT classname, instancename, 
                (SELECT count(*) FROM module AS existing WHERE existing.instancename = newmodules.instancename), 
                config FROM newmodules;
        """)

        # the plugin base is rather easy.
        cur.executemany("INSERT OR IGNORE INTO plugin VALUES (?, ?, ?);", ((plugin, edm[0], edm[1]) for plugin, edm in plugininfo.items()))
        # for the sequence we first insert, then query for the ID, then insert the modules into the relation table.
        cur.execute(f"INSERT OR FAIL INTO sequence({SEQFIELDS}) VALUES({SEQPLACEHOLDER});", (seq))
        seqid = list(cur.execute(f"SELECT id FROM sequence WHERE ({SEQFIELDS}) = ({SEQPLACEHOLDER});", (seq)))
        seqid = seqid[0][0]
        cur.executemany("INSERT INTO sequencemodule SELECT id, ? FROM module WHERE config = ?;", ((seqid, modconfig[label]) for label in modconfig))
        cur.execute("COMMIT;")

def storeworkflows(seqs):
    with sqlite3.connect(DBFILE) as db:
        cur = db.cursor()
        cur.execute("BEGIN;")
        cur.executescript(DBSCHEMA)
        pairs = [[wf] + list(seq) for wf, seqlist in seqs.items() for seq in seqlist]
        cur.executemany(f"INSERT OR IGNORE INTO workflow SELECT ?, (SELECT id FROM sequence WHERE ({SEQFIELDS}) = ({SEQPLACEHOLDER}));", pairs)
        cur.execute("COMMIT;")

def inspectworkflows(wfnumber):
    # here, we run runTheMatrix and then parse the cmsDriver command lines.
    # Not very complicated, but a bit of work.

    # Collect the workflow number where we detected each sequence here, so we can
    # put this data into the DB later.
    sequences = defaultdict(list)

    if wfnumber:
        stepdump = subprocess.check_output(["runTheMatrix.py", "-l", str(wfnumber), "-ne"])
    else:
        stepdump = subprocess.check_output(["runTheMatrix.py", "-ne"])

    lines = stepdump.splitlines()
    workflow = ""
    workflowre = re.compile(b"^([0-9]+.[0-9]+) ")
    for line in lines:
        # if it is a workflow header: save the number.
        m = workflowre.match(line)
        if m:
            workflow = m.group(1).decode()
            continue

        # else, we only care about cmsDriver commands.
        if not b'cmsDriver.py' in line: continue

        args = list(reversed(line.decode().split(" ")))
        step = ""
        scenario = ""
        era = ""
        mc = False
        data = False
        fast = False
        while args:
            item = args.pop()
            if item == '-s':
                step = args.pop()
            if item == '--scenario':
                scenario = args.pop()
            if item == '--era':
                era = args.pop()
            if item == '--data':
                data = True
            if item == '--mc':
                mc = True
            if item == '--fast':
                fast = True
        steps = step.split(",")
        for step in steps:
            s = step.split(":")[0]
            if s in RELEVANTSTEPS:
                # Special case for the default sequence, which is noted as "STEP", not "STEP:".
                if ":" in step:
                    seqs = step.split(":")[1]
                    for seq in seqs.split("+"):
                        sequences[workflow].append(Sequence(seq, s, era, scenario, mc, data, fast))
                else:
                    sequences[workflow].append(Sequence("", s, era, scenario, mc, data, fast))
    return sequences

def processseqs(seqs):
    # launch one map_async per element to get finer grain tasks
    tasks = [stp.map_async(lambda seq: (seq, inspectsequence(seq)), [seq]) for seq in seqs]

    # then watch te progress and write to DB as results become available.
    # That way all the DB access is single-threaded but in parallel with the analysis.
    while tasks:
        time.sleep(1)
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
                storesequenceinfo(seq, *res)
        tasks = running


# A small HTML UI built around http.server. No dependencies!
def serve():
    import traceback
    import http.server

    db = sqlite3.connect(DBFILE)

    def formatseq(seq):
        return (seq.step + ":" + seq.seqname + " " + seq.era + " " + seq.scenario 
            + (" --mc" if seq.mc else "") + (" --data" if seq.data else "") 
            + (" --fast" if seq.fast else ""))

    def index():
        out = []
        cur = db.cursor()
        out.append("<H2>Sequences</H2><ul>")
        out.append("""<p> A sequence name, given as <em>STEP:@sequencename</em> here, does not uniquely identify a sequence.
            The modules on the sequence might depend on other cmsDriver options, such as Era, Scenario, Data vs. MC, etc.
            This tool lists parameter combinations that were observed. However, sequences with identical contents are grouped
            on this page. The default sequence, used when no explicit sequence is apssed to cmsDriver, is noted as <em>STEP:</em>.</p>""")
        rows = cur.execute(f"SELECT seqname, step, count(*) FROM sequence GROUP BY seqname, step ORDER BY seqname, step;")
        for row in rows:
            seqname, step, count = row
            out.append(f' <li>')
            out += showseq(step, seqname)
            out.append(f' </li>')
        out.append("</ul>")

        out.append("<H2>Modules</H2><ul>")
        rows = cur.execute(f"SELECT classname, edmfamily, edmbase FROM plugin ORDER BY edmfamily, edmbase, classname")
        for row in rows:
            classname, edmfamily, edmbase = row
            if not edmfamily: edmfamily = "<em>legacy</em>"
            out.append(f' <li>{edmfamily}::{edmbase} <a href="/plugin/{classname}/">{classname}</a></li>')
        out.append("</ul>")
        return out

    def showseq(step, seqname):
        # display set of sequences sharing a name, also used on the index page.
        out = []
        cur = db.cursor()
        out.append(f'     <a href="/seq/{step}:{seqname}/">{step}:{seqname}</a>')
        # this is much more complicated than it should be since we don't keep
        # track which sequences have equal contents in the DB. So the deduplication
        # has to happen in Python code.
        rows = cur.execute(f"SELECT {SEQFIELDS}, moduleid, id    FROM sequence INNER JOIN sequencemodule ON sequenceid = id WHERE seqname = ? and step = ?;", (seqname, step))

        seqs = defaultdict(list)
        ids = dict()
        for row in rows:
            seq = Sequence(*row[:-2])
            seqs[seq].append(row[-2])
            ids[seq] = row[-1]

        variations = defaultdict(list)
        for seq, mods in seqs.items():
            variations[tuple(sorted(mods))].append(seq)

        out.append("        <ul>")
        for mods, seqs in variations.items():
            count = len(mods)
            out.append(f'            <li>({count} modules):')
            for seq in seqs:
                seqid = ids[seq]
                out.append(f'<br><a href="/seqid/{seqid}">' + formatseq(seq) + '</a>')
                # This query in a loop is rather slow, but this got complictated enough, so YOLO.
                rows = cur.execute("SELECT wfid FROM workflow WHERE sequenceid = ?;", (seqid,))
                out.append(f'<em>Used on workflows: ' + ", ".join(wfid for wfid, in rows) + "</em>")
            out.append('            </li>')
        out.append("        </ul>")
        return out

    def showseqid(seqid):
        # display a single, unique sequence.
        seqid = int(seqid)
        out = []
        cur = db.cursor()
        rows = cur.execute(f"SELECT {SEQFIELDS} FROM sequence WHERE id = ?;", (seqid,))
        seq = formatseq(Sequence(*list(rows)[0]))
        out.append(f"<h2>Modules on {seq}:</h2><ul>")
        rows = cur.execute("SELECT wfid FROM workflow WHERE sequenceid = ?;", (seqid,))
        out.append("<p><em>Used on workflows: " + ", ".join(wfid for wfid, in rows) + "</em></p>")
        rows = cur.execute("""
            SELECT classname, instancename, variation, moduleid    
            FROM sequencemodule INNER JOIN module ON moduleid = module.id
            WHERE sequenceid = ?;""", (seqid,))
        for row in rows:
            classname, instancename, variation, moduleid = row
            out.append(f'<li>{instancename} ' + (f'<sub>{variation}</sub>' if variation else '') + f' : <a href="/plugin/{classname}/">{classname}</a></li>')
        out.append("</ul>")

        return out

    def showclass(classname):
        # display all known instances of a class and where they are used.
        # this suffers a bit from the fact that fully identifying a sequence is 
        # rather hard, we just show step/name here.
        out = []
        out.append(f"<h2>Plugin {classname}</h2>")
        cur = db.cursor()
        # First, info about the class iself.
        rows = cur.execute("SELECT edmfamily, edmbase FROM plugin WHERE classname = ?;", (classname,))
        edmfamily, edmbase = list(rows)[0]
        islegcay = not edmfamily
        if islegcay: edmfamily = "<em>legacy</em>"
        out.append(f"<p>{classname} is a <b>{edmfamily}::{edmbase}</b>.</p>")
        out.append("""<p>A module with a given label can have different configuration depending on options such as Era,
            Scenario, Data vs. MC etc. If multiple configurations for the same name were found, they are listed separately
            here and denoted using subscripts.</p>""")
        if (edmbase != "EDProducer" and not (islegcay and edmbase == "EDAnalyzer")) or (islegcay and edmbase == "EDProducer"):
            out.append(f"<p>This is not a DQM module.</p>")

        # then, its instances.
        rows = cur.execute("""
            SELECT module.id, instancename, variation, sequenceid, step, seqname 
            FROM module INNER JOIN sequencemodule ON moduleid = module.id INNER JOIN sequence ON sequence.id == sequenceid
            WHERE classname = ? ORDER BY instancename, variation, step, seqname;""", (classname,))
        out.append("<ul>")
        seqsformod = defaultdict(list)
        liformod = dict()
        for row in rows:
            id, instancename, variation, sequenceid, step, seqname = row
            liformod[id] = f'<a href="/config/{id}">{instancename}' + (f"<sub>{variation}</sub>" if variation else '') + "</a>"
            seqsformod[id].append((sequenceid, f"{step}:{seqname}"))
        for id, li in liformod.items():
            out.append("<li>" + li + ' Used here: ' + ", ".join(f'<a href="/seqid/{seqid}">{name}</a>' for seqid, name in seqsformod[id]) + '.</li>')
        out.append("</ul>")
        return out

    def showconfig(modid):
        # finally, just dump the config of a specific module. Useful to do "diff" on it.
        modid = int(modid)
        out = []
        cur = db.cursor()
        rows = cur.execute(f"SELECT config FROM module WHERE id = ?;", (modid,))
        config = list(rows)[0][0]
        out.append("<pre>")
        out.append(config.decode())
        out.append("</pre>")
        return out

    ROUTES = [
        (re.compile('/$'), index),
        (re.compile('/seq/(\w+):([@\w]*)/$'), showseq),
        (re.compile('/seqid/(\d+)$'), showseqid),
        (re.compile('/config/(\d+)$'), showconfig),
        (re.compile('/plugin/(.*)/$'), showclass),
    ]

    # the server boilerplate.
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            try:
                res = None
                for pattern, func in ROUTES:
                    m = pattern.match(self.path)
                    if m:
                        res = "\n".join(func(*m.groups())).encode("utf8")
                        break

                if res:
                    self.send_response(200, "Here you go")
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b"""<html><style>
                        body {
                            font-family: sans;
                        }
                    </style><body>""")
                    self.wfile.write(res)
                    self.wfile.write(b"</body></html>")
                else:
                    self.send_response(400, "Something went wrong")
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b"I don't understand this request.")
            except:
                trace = traceback.format_exc()
                self.send_response(500, "Things went very wrong")
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(trace.encode("utf8"))

    server_address = ('', 8000)
    httpd = http.server.HTTPServer(server_address, Handler)
    print("Serving at http://localhost:8000/ ...")
    httpd.serve_forever()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Collect information about DQM sequences.')
    parser.add_argument("--sequence", default="", help="Name of the sequence")
    parser.add_argument("--step", default="DQM", help="cmsDriver step that the sequence applies to")
    parser.add_argument("--era", default="Run2_2018", help="CMSSW Era to use")
    parser.add_argument("--scenario", default="pp", help="cmsDriver scenario")
    parser.add_argument("--data", default=False, action="store_true", help="Pass --data to cmsDriver.")
    parser.add_argument("--mc", default=False, action="store_true", help="Pass --mc to cmsDriver.")
    parser.add_argument("--fast", default=False, action="store_true", help="Pass --fast to cmsDriver.")
    parser.add_argument("--workflow", default=None, help="Ignore other options and inspect this workflow instead (implies --sqlite).")
    parser.add_argument("--runTheMatrix", default=False, action="store_true", help="Ignore other options and inspect the full matrix instea (implies --sqlite).")
    parser.add_argument("--steps", default="ALCA,ALCAPRODUCER,ALCAHARVEST,DQM,HARVESTING,VALIDATION", help="Which workflow steps to inspect from runTheMatrix.")
    parser.add_argument("--sqlite", default=False, action="store_true", help="Write information to SQLite DB instead of stdout.")
    parser.add_argument("--dbfile", default="sequences.db", help="Name of the DB file to use.")
    parser.add_argument("--infile", default=INFILE, help="LFN/PFN of input file to use. Default is %s" % INFILE)
    parser.add_argument("--threads", default=None, type=int, help="Use a fixed number of threads (default is #cores).")
    parser.add_argument("--limit", default=None, type=int, help="Process only this many sequences.")
    parser.add_argument("--offset", default=None, type=int, help="Process sequences starting from this index. Used with --limit to divide the work into jobs.")
    parser.add_argument("--showpluginlabel", default=False, action="store_true", help="Print the module label for each plugin (default).")
    parser.add_argument("--showplugintype", default=False, action="store_true", help="Print the base class for each plugin.")
    parser.add_argument("--showpluginclass", default=False, action="store_true", help="Print the class name for each plugin.")
    parser.add_argument("--showpluginconfig", default=False, action="store_true", help="Print the config dump for each plugin.")
    parser.add_argument("--serve", default=False, action="store_true", help="Ignore other options and instead serve HTML UI from SQLite DB.")

    args = parser.parse_args()

    RELEVANTSTEPS += args.steps.split(",")
    DBFILE = args.dbfile

    if args.threads:
      tp = ThreadPool(args.threads)
      stp = ThreadPool(args.threads)

    INFILE = args.infile
    if args.serve:
        serve()
    elif args.workflow or args.runTheMatrix:
        # the default workflow None is a magic value for inspectworkflows.
        seqs = inspectworkflows(args.workflow)
        seqset = set(sum(seqs.values(), []))
        if args.offset:
            seqset = list(sorted(seqset))[args.offset:]
        if args.limit:
            seqset = list(sorted(seqset))[:args.limit]

        print("Analyzing %d seqs..." % len(seqset))

        processseqs(seqset)
        storeworkflows(seqs)
    else:
        # single sequence with arguments from commandline...
        seq = Sequence(args.sequence, args.step, args.era, args.scenario, args.mc, args.data, args.fast)
        modconfig, modclass, plugininfo = inspectsequence(seq)
        if args.sqlite:
            storesequenceinfo(seq, modconfig, modclass, plugininfo)
        else:
            # ... and output to stdout.
            if not (args.showpluginlabel or args.showpluginclass or args.showplugintype or args.showpluginconfig):
                args.showpluginlabel = True
            formatsequenceinfo(modconfig, modclass, plugininfo, args.showpluginlabel, args.showpluginclass, args.showplugintype,    args.showpluginconfig)
