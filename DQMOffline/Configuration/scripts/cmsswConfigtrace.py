#!/usr/bin/env python3
from __future__ import print_function
import os
import re
import six
import sys
import inspect
import sqlite3
import tempfile
import traceback
import subprocess
from shutil import copy, rmtree
from collections import defaultdict

### 1. Some configuration options.
SERVE_PORT = 1234

OUTFILE_TREE = "configtree.sqlite"
IGNORE_PACKAGES = ['FWCore/ParameterSet', 'DQMOffline/Configuration/scripts', "cmsRun"]
STRIPPATHS = [ # we will add the base dir from CMSSWCALLBASE env var here
  os.environ["CMSSW_BASE"] + "/python/", os.environ["CMSSW_RELEASE_BASE"] + "/python/",
  os.environ["CMSSW_BASE"] + "/cfipython/", os.environ["CMSSW_RELEASE_BASE"] + "/cfipython/"]
PREFIXINFO = [] # what we will show as "source" later
ARGV0 = "" # set in main

### 2. The enable all the tracing hooks.

# this already does a good job, but it is not enough
#import FWCore.GuiBrowsers.EnablePSetHistory

# we need to patch out this to allow printing unlabled things
import FWCore.ParameterSet.Mixins
del FWCore.ParameterSet.Mixins._Labelable.__str__

# then also trace Sequence construction so we can get a full tree
# (PSetHistory only tracks leaves)
def auto_inspect():
  # stolen from FWCore.GuiBrowsers.EnablePSetHistory, but needs more black-list
  stack = inspect.stack()
  i = 0
  while i < len(stack) and len(stack[i])>=2 and any(map(lambda p: p in stack[i][1], IGNORE_PACKAGES)):
    i += 1
  res = stack[i: ]
  j = 0
  # for the other end we only use cmsRun (instead of IGNORE_PACKAGES) to avoid
  # cutting traces in the middle that enter and leave IGNORE_PACKAGES
  while j < len(res) and not 'cmsRun' in res[j][1]:
    j += 1
  res = res[:j]
  if len(res)>=1 and len(res[0])>=3:
    return res
  else:
    return [("unknown","unknown","unknown")]

def trace_location(thing, name, extra = lambda thing, *args, **kwargs: thing):
  old_method = getattr(thing, name)
  def trace_location_hook(self, *args, **kwargs):
    retval = old_method(self, *args, **kwargs)
    where = auto_inspect()
    #print("Called %s::%s at %s" % (thing.__name__, name, where[0][1:3]))
    event = (name, tuple(w[1:3] for w in where), extra(self, *args, **kwargs))
    if hasattr(self, "_trace_events"):
      getattr(self, "_trace_events").append(event)
    else:
      # this bypasses setattr checks
      self.__dict__["_trace_events"] = [ event ]

    return retval
  setattr(thing, name, trace_location_hook)

def flatten(*args):
  # that was surprisingly hard...
  return     [x  for x in args if not isinstance(x, list)] + sum(
    [flatten(*x) for x in args if isinstance(x, list)], [])

from FWCore.ParameterSet.SequenceTypes import _ModuleSequenceType, _SequenceCollection, Task, _UnarySequenceOperator, Schedule
from FWCore.ParameterSet.Modules import _Module, Source, ESSource, ESPrefer, ESProducer, Service, Looper
from FWCore.ParameterSet.Config import Process
# with this we can also track the '+' and '*' of modules, but it is slow
trace_location(_SequenceCollection, '__init__')
trace_location(_Module, '__init__')
trace_location(Source, '__init__')
trace_location(ESSource, '__init__')
trace_location(ESPrefer, '__init__')
trace_location(ESProducer, '__init__')
trace_location(Service, '__init__')
trace_location(Looper, '__init__')
trace_location(Process, '__init__')
trace_location(_UnarySequenceOperator, '__init__')
# lambda agrument names all match the original declarations, to make kwargs work
trace_location(_ModuleSequenceType, '__init__', lambda self, *arg: {'args': list(arg)})
trace_location(_ModuleSequenceType, 'copy')
trace_location(_ModuleSequenceType, 'associate', lambda self, *tasks: {'args': list(tasks)})
trace_location(_ModuleSequenceType, '__imul__', lambda self, rhs: {'rhs': rhs})
trace_location(_ModuleSequenceType, '__iadd__', lambda self, rhs: {'rhs': rhs})
trace_location(_ModuleSequenceType, 'copyAndExclude', lambda self, listOfModulesToExclude: {'olds': list(listOfModulesToExclude)})
trace_location(_ModuleSequenceType, 'replace', lambda self, original, replacement: {'old': original, 'new': replacement})
trace_location(_ModuleSequenceType, 'insert', lambda self, index, item: {'rhs': item})
trace_location(_ModuleSequenceType, 'remove', lambda self, something: {'old': something})
trace_location(Task, '__init__')
trace_location(Task, 'add', lambda self, *items: {'args': list(items)})
trace_location(Task, 'copy')
trace_location(Task, 'copyAndExclude', lambda self, listOfModulesToExclude: {'olds': list(listOfModulesToExclude)})
trace_location(Schedule, '__init__', lambda self, *args, **kwargs: {'args': flatten(list(args), kwargs.values())})
trace_location(Schedule, 'associate', lambda self, *tasks: {'args': list(tasks)})
trace_location(Schedule, 'copy')
# TODO: we could go deeper into Types and PSet, but that should not be needed for now.

# lifted from EnablePSetHistory, we don't need all of that stuff.
def new_items_(self):
  items = []
  if self.source:
    items += [("source", self.source)]
  if self.looper:
    items += [("looper", self.looper)]
  #items += self.moduleItems_()
  items += self.outputModules.items()
  #items += self.sequences.items() # TODO: we don't need sequences that are not paths?
  items += six.iteritems(self.paths)
  items += self.endpaths.items()
  items += self.services.items()
  items += self.es_producers.items()
  items += self.es_sources.items()
  items += self.es_prefers.items()
  #items += self.psets.items()
  #items += self.vpsets.items()
  if self.schedule:
    items += [("schedule", self.schedule)]
  return tuple(items)
Process.items_=new_items_

### 3. The logic to collect, process, and save the information from the process.

def collect_trace(thing, name, graph, parent):
  # thing is what to look at, graph is the output list (of child, parent tuple pairs)
  # thing could be pretty much anything.
  classname = thing.__class__.__name__
  if hasattr(thing, '_trace_events'):
    events = list(getattr(thing, '_trace_events'))
    getattr(thing, '_trace_events')[:] = [] # erase events so we can't end up in cycles
    for action, loc, extra in events:
      entry = (action, classname, loc)
      graph.append((entry, parent, name))

      # items shall be a list of tuples (type, object) of the immediate children of the thing.
      items = []
      if hasattr(extra, 'items_'): # for cms.Process
        items += extra.items_()
      if hasattr(extra, '_seq'): # for sequences and similar
        seq = getattr(extra, '_seq')
        if seq:
          items += [('seqitem', x) for x in getattr(seq, '_collection')]
      if hasattr(extra, '_tasks'): # same
        items += [('task', x) for x in getattr(extra, '_tasks')]
      if hasattr(extra, '_collection'): # for cms.Task
        items += [('subtask', x) for x in getattr(extra, '_collection')]
      if hasattr(extra, '_operand'): # for _SeqenceNegation etc.
        items += [('operand', getattr(extra, '_operand'))]
      if isinstance(extra, dict): # stuff that we track explicitly^
        for key, value in extra.items():
          if isinstance(value, list):
            items += [(key, x) for x in value]
          else:
            items += [(key, value)]

      for name, child in items:
        collect_trace(child, name, graph, entry)

  else:
    if not thing is None:
      print("No _trace_events found in %s.\nMaybe turn on tracing for %s?" % (thing, classname))
      print(" Found in %s" % (parent,))


def writeoutput(graph):
  progname = " ".join(PREFIXINFO)
  print("+Done running %s, writing output..." % progname)

  def formatfile(filename):
    filename = os.path.abspath(filename)
    for pfx in STRIPPATHS:
      if filename.startswith(pfx):
        filename = filename[len(pfx):]
    return filename

  files = set()
  for child, parent, relation in graph:
    files.add(child[2][0][0])
    files.add(parent[2][0][0])

  conn = sqlite3.connect(os.environ["CMSSWCALLTREE"])
  cur = conn.cursor()
  cur.executescript("""
  CREATE TABLE IF NOT EXISTS file(id INTEGER PRIMARY KEY, 
    name TEXT, UNIQUE(name)
  );
  CREATE TABLE IF NOT EXISTS trace(id INTEGER PRIMARY KEY, 
    parent INTEGER, -- points into same table, recursively
    file INTEGER, line INTEGER,
    FOREIGN KEY(parent) REFERENCES trace(id),
    FOREIGN KEY(file) REFERENCES file(id),
    UNIQUE(parent, file, line)
  );
  CREATE TABLE IF NOT EXISTS relation(id INTEGER PRIMARY KEY,
    place  INTEGER, 
    place_type TEXT,
    usedby INTEGER, 
    usedby_type TEXT,
    relation TEXT,
    source TEXT,
    FOREIGN KEY(place) REFERENCES trace(id),
    FOREIGN KEY(usedby) REFERENCES trace(id)
  );
  CREATE INDEX IF NOT EXISTS placeidx ON relation(place);
  CREATE INDEX IF NOT EXISTS usedbyidx ON relation(usedby);
  CREATE INDEX IF NOT EXISTS traceidx ON trace(file);
  -- SQLite does not optimise that one well, but a VIEW is still nice to have...
  CREATE VIEW IF NOT EXISTS fulltrace AS 
    WITH RECURSIVE fulltrace(level, baseid, parent, file, name, line) AS (
      SELECT 1 AS level, trace.id, parent, trace.file, file.name, line FROM trace
        INNER JOIN file ON file.id = trace.file
      UNION SELECT level+1, baseid, trace.parent, trace.file, file.name, trace.line FROM fulltrace 
        INNER JOIN trace ON trace.id = fulltrace.parent
        INNER JOIN file ON file.id = trace.file)
    SELECT * FROM fulltrace;
  """)
  cur.executemany("INSERT OR IGNORE INTO file(name) VALUES (?);", 
    ((formatfile(f),) for f in files))
  def inserttrace(loc):
    parent = 0
    for filename, line in reversed(loc):
      conn.execute("INSERT OR IGNORE INTO trace(parent, file, line) SELECT ?, id, ? FROM file WHERE name == ?;", (parent, line, formatfile(filename)))
      cur = conn.execute("SELECT trace.id FROM trace LEFT JOIN file ON trace.file == file.id WHERE trace.parent = ? AND file.name = ? AND trace.line = ?;", (parent, formatfile(filename), line))
      for row in cur:
        parent = row[0]
    return parent

  for child, parent, relation in graph:
    cevt, cclassname, cloc = child
    pevt, pclassname, ploc = parent
    place = inserttrace(cloc)
    usedby = inserttrace(ploc)
    cur.execute("INSERT OR IGNORE INTO relation(place, place_type, usedby, usedby_type, relation, source) VALUES (?,?,?,?,?,?);", (
      place, "%s::%s" % (cclassname, cevt),
      usedby, "%s::%s" % (pclassname, pevt),
      relation, progname
    ))
  conn.commit()
  conn.close()

### 4. Launch and keep track of all the processes.

def addprefixinfo(argv):
  cwd = os.path.abspath(os.getcwd())
  wf = re.match(".*/(\d+\.\d+)_", cwd)
  if wf: 
    PREFIXINFO.append("wf")
    PREFIXINFO.append(wf.groups()[0])
  online = re.match("(.*/)?(.*)_dqm_sourceclient-live_cfg\.py", argv[0])
  if online:
    PREFIXINFO.append("online")
    PREFIXINFO.append(online.groups()[1])
  step = re.match("(step\d+)_.*\.py", argv[0])
  if step:
    PREFIXINFO.append(step.groups()[0])
  processing = re.match("step\d+_.*(RECO|ALCA|HARVEST).*\.py", argv[0])
  if processing:
    PREFIXINFO.append(processing.groups()[0])
  if not PREFIXINFO:
    PREFIXINFO.append(argv[0])

def setupenv():
  bindir = tempfile.mkdtemp()
  print("+Setting up in ", bindir)
  os.symlink(ARGV0, bindir + "/cmsRun")
  os.environ["PATH"] = bindir + ":" + os.environ["PATH"]
  os.environ["CMSSWCALLTREE"] = bindir + "/" + OUTFILE_TREE
  os.environ["CMSSWCALLBASE"] = os.path.abspath(os.getcwd()) + "/"
  with open(os.environ["CMSSWCALLTREE"], "w") as f:
    pass
  return bindir

def cleanupenv(tmpdir):
  print("+Cleaning up ", tmpdir)
  copy(os.environ["CMSSWCALLTREE"], ".")
  rmtree(tmpdir)


def trace_command(argv):
  tmpdir = None
  if not "CMSSWCALLTREE" in os.environ:
    tmpdir = setupenv()

  subprocess.call(argv)

  if tmpdir:
    cleanupenv(tmpdir)

def trace_python(prog_argv, path):
  graph = []
  sys.argv = prog_argv
  progname = prog_argv[0]
  file_path = searchinpath(progname, path)
  try:
    with open(file_path) as fp:
      code = compile(fp.read(), progname, 'exec')
      globals = {}
      try:
        exec code in globals, globals
      except:
        print(traceback.format_exc())
      finally:
        # reporting is only possible if the config was executed successfully.
        # we still do it in case of an exception, which can happen after convertToUnscheduled()
        print("+Collecting trace information from %s..." % globals["process"])
        collect_trace(globals["process"], 'cmsrun', graph, ('cmsRun', '', ((progname, 0),)))
        writeoutput(graph)

  except OSError as err:
    print("+Cannot run file %r because: %s" % (sys.argv[0], err))
    sys.exit(1)
  except SystemExit:
    pass
  # this is not necessarily reached at all. 
  sys.exit(0)

def searchinpath(progname, path):
  # Search $PATH. There seems to be no pre-made function for this.
  for entry in path:
    file_path = os.path.join(entry, progname)
    if os.path.isfile(file_path):
      break
  if not os.path.isfile(file_path):
    print("+Cannot find program (%s) in modified $PATH (%s)." % (progname, path))
    sys.exit(1)
  print("+Found %s as %s in %s." % (progname, file_path, path))
  return file_path


def help():
  print("Usage: %s <some cmssw commandline>" % (sys.argv[0]))
  print("  The given programs will be executed, instrumenting calls to cmsRun.")
  print("  cmsRun will not actually run cmssw, but all the Python code will be executed and instrumentd. The results are written to the file `%s` in the same directory." % OUTFILE_TREE)
  print("  The callgraph output lists edges pointing from each function to the one calling it.")
  print("  To view the results using a simple webpage, use\n   %s serve" % sys.argv[0])
  print("Examples:")
  print("  %s runTheMatrix.py -l 1000 --ibeos" % sys.argv[0])

def main():
  print("+Running cmsswConfigtrace...")
  global ARGV0
  ARGV0 = sys.argv[0]
  if sys.argv[0].endswith('cmsRun'):
      print("+Wrapping cmsRun...")
      addprefixinfo(sys.argv[1:])
      STRIPPATHS.append(os.environ["CMSSWCALLBASE"])
      trace_python(sys.argv[1:], ["."])
      return
  if len(sys.argv) <= 1:
    help()
    return
  # else
  print("+Running command with tracing %s..." % sys.argv[1:])
  trace_command(sys.argv[1:])

### 5. The web-based GUI.

def serve_main():
  # we use STRIPPATHS as a search path to find code files.
  STRIPPATHS.append(os.path.abspath(os.getcwd()) + "/")

  import SimpleHTTPServer
  import SocketServer
  import urllib 

  conn = sqlite3.connect(OUTFILE_TREE)

  def escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

  def formatsource(source, formatstr = '<em class="%s">%s</em>'):
    processings = ["ALCA", "RECO", "HARVEST", "online"]
    info = source.split(" ")
    processing = " ".join(filter(lambda x: x in processings, info))
    source = " ".join(filter(lambda x: x not in processings, info))
    return formatstr % (processing, escape(source))

  def formatplace(filename, line):
    MAXLEN = 80
    shortname = filename[ :(MAXLEN-3)/2] + "..." + filename[-(MAXLEN-3)/2: ] if len(filename) > MAXLEN else filename
    return '<a href="/%s#%s" target="_blank">%s:%s</a>' % (escape(filename), line, escape(shortname), line)

  def index():
    out = [escape('goto to /<filename> for info about a file')]

    wfs = defaultdict(list)
    for source, wf in conn.execute(""" -- some SQL hackery here to parse "source"
        SELECT DISTINCT source, substr(source, instr(source, "wf ")+3)*1 FROM relation ORDER BY 2, 1;"""):
      wfs[wf].append('<a href="/workflow/%s">%s</a>' % (urllib.quote(source), formatsource(source)))
    out.append("<ul>")
    for wf in wfs:
      out.append('<li>' + " ".join(wfs[wf]) + "</li>")
    out.append("</ul>")

    out.append("<ul>")
    for f in conn.execute("SELECT name FROM file ORDER BY name;"):
      name = escape(f[0])
      out.append('<li><a href="/%s">%s</a></li>' % (name, name))
    out.append("</ul>")

    return "\n".join(out)

  def showworkflow(source):
    source = urllib.unquote(source)
    cur = conn.execute("""
      SELECT DISTINCT file.name FROM relation 
        INNER JOIN trace ON place = trace.id
        INNER JOIN file ON file.id = trace.file
      WHERE relation.source = ?
      ORDER BY file.name;
    """, (source, ))
    out = ["Files used by workflow %s: <ul>" % formatsource(source)]
    for f in cur:
      name = escape(f[0])
      out.append('<li><a href="/%s">%s</a></li>' % (name, name))
    out.append("</ul>")
    return "\n".join(out)

  def showfile(filename):
    out = []
    out.append('<script src="https://rawgit.com/google/code-prettify/master/src/prettify.js"></script>')
    out.append('<link rel="stylesheet" href="https://rawgit.com/google/code-prettify/master/src/prettify.css"></link>')

    lines = None
    for d in STRIPPATHS:
      try:
        with open(d + filename) as f:
          lines = f.readlines()
          out.append("Read %s" % f.name)
        break
      except:
        pass

    if lines:
      cur = conn.execute("""
      SELECT DISTINCT trace.line, source FROM relation 
        INNER JOIN trace on relation.place = trace.id 
        INNER JOIN file ON trace.file == file.id 
        WHERE file.name == ? ORDER BY line, source;""", (filename,))
      sourceinfo = defaultdict(list)
      for line, source in cur:
        sourceinfo[line].append(source)

      out.append('<pre class="prettyprint linenums">')
      for i, l in enumerate(lines):
        # put the text into data-tag here and move it later, to not mess up syntax HL
        tags = [formatsource(source, '<em class="%%s" data-tag="%%s" data-line="%d"></em>' % (i+1)) for source in sourceinfo[i+1]]
        out.append(escape(l).rstrip() + "".join(tags))
      out.append('</pre>')

      out.append("""<script type="text/javascript">
      PR.prettyPrint();
      clickfunc = function(evt) {
        document.querySelectorAll("li > iframe, li > br").forEach(function(e) {e.remove()}); 
        dest = "/info" + window.location.pathname + ":" + this.getAttribute("data-line");
        this.parentElement.insertAdjacentHTML("beforeend", '<br><iframe width="90%" height="500px" frameborder="0" src="' + dest + '"></iframe><br>');
      };
      document.querySelectorAll("li > em").forEach(function(e) {
        e.innerText = e.getAttribute("data-tag");
        e.onclick = clickfunc;
      })

      n = 1*window.location.hash.replace("#", "");
      if (n > 0) {
        li = document.querySelectorAll("li")[n-1];
        li.style = "background: #ee7";
        li.scrollIntoView();
      }
      </script>""")

    else:
      out.append("Could not find %s" % filename)

    return "\n".join(out)

      
  def showinfo(filename, line):
    line = int(line)
    out = []
    def queryandoutput(startfrom, to, directiontext):
      # we format in the side of the relation to query for here...
      cur = conn.execute("""
        SELECT place_type, -- why did we trace this line?
            <to>file.name, <to>trace.line, -- where was it used?
            usedby, usedby_type, relation, -- what was it used for?
            place, source -- why did this code run?
          FROM relation 
          INNER JOIN trace AS placetrace  ON placetrace.id  = relation.place
          INNER JOIN trace AS usedbytrace ON usedbytrace.id = relation.usedby
          INNER JOIN file AS placefile  ON placefile.id  = placetrace.file
          INNER JOIN file AS usedbyfile ON usedbyfile.id = usedbytrace.file
        WHERE <from>file.name = ? AND <from>trace.line = ?
        LIMIT 1000; """
        .replace("<from>", startfrom).replace("<to>", to), (filename, line))
      out.append("<p>%s %s <ul>" % (formatplace(filename, line), directiontext))
      for place_type, pname, pline, usedby, usedby_type, relation, place, source in cur:
        out.append(
          '<li><tt>%s</tt> at %s by <tt>%s</tt> as <a href="/why/%d">%s</a> <a href="/why/%d">in</a> %s</li>'
          % (escape(place_type), formatplace(pname, pline), escape(usedby_type), usedby, escape(relation), place, formatsource(source)))
      out.append("</ul></p>")

    queryandoutput("place", "usedby", "is used as")
    queryandoutput("usedby", "place", "uses")

    return "\n".join(out)

  def showwhy(id):
    id = int(id)
    # this (WHERE before recursion) will optimize better than the view.
    cur = conn.execute("""
      WITH RECURSIVE fulltrace(level, baseid, parent, file, name, line) AS (
        SELECT 1 AS level, trace.id, parent, trace.file, file.name, line FROM trace 
          INNER JOIN file ON file.id = trace.file
        WHERE trace.id = ?
        UNION SELECT level+1, baseid, trace.parent, trace.file, file.name, trace.line FROM fulltrace 
          INNER JOIN trace ON trace.id = fulltrace.parent
          INNER JOIN file ON file.id = trace.file)
      SELECT name, line FROM fulltrace ORDER BY level;""", (id,))
    out = []
    out.append("Full stack trace:<ul>")
    for name, line in cur:
      out.append('<li>%s</li>' % formatplace(name, line))
    out.append("</ul>")
    return "\n".join(out)

  ROUTES = [
    (re.compile('/workflow/(.*)$'), showworkflow),
    (re.compile('/info/(.*):(\d+)$'), showinfo),
    (re.compile('/why/(\d+)$'), showwhy),
    (re.compile('/([^.]*[.]?[^.]+[.]?[^.]*)$'), showfile),
    (re.compile('/$'), index),
  ]

  def do_GET(self):
    try:
      res = None
      for pattern, func in ROUTES:
        m = pattern.match(self.path)
        if m:
          res = func(*m.groups())
          break

      if res:
        self.send_response(200, "Here you go")
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write("""<html><style>
          body {
            font-family: sans;
          }
          em {
            cursor: pointer;
            padding: 0 2px;
            margin: 1 2px;
            background: #999;
          }
          em.ALCA {background: #ee9; }
          em.RECO {background: #9e9; }
          em.HARVEST {background: #99e; }
          em.online {background: #e99; }
        </style><body>""")
        self.wfile.write(res)
        self.wfile.write("</body></html>")
        self.wfile.close()
      else:
        self.send_response(400, "Something went wrong")
    except:
      trace = traceback.format_exc()
      self.send_response(500, "Things went very wrong")
      self.send_header("Content-Type", "text/plain; charset=utf-8")
      self.end_headers()
      self.wfile.write(trace)
      self.wfile.close()

  Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
  Handler.do_GET = do_GET
  httpd = SocketServer.TCPServer(("",SERVE_PORT), Handler)
  print("serving at port", SERVE_PORT)
  httpd.serve_forever()

if __name__ == '__main__':
  if sys.argv[1] == "serve":
    serve_main()
  else:
    main()
  
