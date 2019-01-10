#!/usr/bin/env python2
from __future__ import print_function
import os
import six
import sys
import inspect
import tempfile
import subprocess
from shutil import copy, rmtree
from collections import defaultdict

import FWCore
import FWCore.Modules.EmptySource_cfi

OUTFILE_TREE = "calltree"
def STRIPPATHS():
  # this can only be evaluated after FWCore is loaded, and instrumented
  return [
    "/".join(os.path.dirname(FWCore.__file__).split("/")[:-1]) + "/",
    "/".join(FWCore.Modules.EmptySource_cfi.__file__.split("/")[:-3]) + "/",
  ]
IGNORE_PACKAGES = ['FWCore/ParameterSet', 'FWCore/GuiBrowsers', 'DQMOffline/Configuration/scripts', "cmsRun"]

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
  if len(res)>=1 and len(res[0])>=3:
    return res
  else:
    return [("unknown","unknown","unknown")]

def trace_location(thing, name):
  old_method = getattr(thing, name)
  def trace_location_hook(self, *args, **kwargs):
    where = auto_inspect()
    #print("Called %s::%s at %s" % (thing.__name__, name, where[0][1:3]))
    event = (name, where[0][1: ])
    if hasattr(self, "_trace_events"):
      getattr(self, "_trace_events").append(event)
    else:
      # this bypasses setattr checks
      self.__dict__["_trace_events"] = [ event ]

    return old_method(self, *args, **kwargs)
  setattr(thing, name, trace_location_hook)

from FWCore.ParameterSet.SequenceTypes import _ModuleSequenceType, _Sequenceable, Task
from FWCore.ParameterSet.Modules import _Module, Source, ESSource, ESPrefer, ESProducer, Service, Looper
from FWCore.ParameterSet.Config import Process
# with this we can also track the '+' and '*' of modules, but it is slow
#trace_location(_Sequenceable, '__init__')
trace_location(_Module, '__init__')
trace_location(Source, '__init__')
trace_location(ESSource, '__init__')
trace_location(ESPrefer, '__init__')
trace_location(ESProducer, '__init__')
trace_location(Service, '__init__')
trace_location(Looper, '__init__')
trace_location(Process, '__init__')
# TODO: things to track:
# - __init__ (place, content is is _seq)
# - associate (track place)
# - __imul__, __iadd__ (track place)
# - copyAndExclude (what was excluded?)
# - replace (track place, removed things)
# - insert (place)
# - remove (place, what?)
trace_location(_ModuleSequenceType, '__init__')
trace_location(_ModuleSequenceType, 'associate')
trace_location(_ModuleSequenceType, '__imul__')
trace_location(_ModuleSequenceType, '__iadd__')
trace_location(_ModuleSequenceType, 'copyAndExclude')
trace_location(_ModuleSequenceType, 'replace')
trace_location(_ModuleSequenceType, 'insert')
trace_location(_ModuleSequenceType, 'remove')
trace_location(Task, '__init__')
trace_location(Task, 'add')
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


def instrument_cmssw():
  # everything happens in the imports so far
  pass

def collect_trace(thing, graph, parent):
  # thing is what to look at, graph is the output list (of child, parent tuple pairs)
  # thing could be pretty much anything.
  classname = thing.__class__.__name__
  if hasattr(thing, '_trace_events'):
    events = getattr(thing, '_trace_events')
    for action, loc in events:
      filename = loc[0]
      line = loc[1]
      entry = (action, classname, filename, line)
      graph.append((entry, parent))
  else:
    print("No _trace_events found in %s.\nMaybe turn on tracing for %s?" % (thing, classname))
    print(" Found in %s" % (parent,))

  # items shall be a list of tuples (type, object) of the immediate children of the thing.
  items = []
  if hasattr(thing, 'items_'): # for cms.Process
    items += thing.items_()
  if hasattr(thing, '_seq'): # for sequences and similar
    seq = getattr(thing, '_seq')
    if seq:
      items += [('seqitem', x) for x in getattr(seq, '_collection')]
  if hasattr(thing, '_tasks'): # same
    items += [('task', x) for x in getattr(thing, '_tasks')]
  if hasattr(thing, '_collection'): # for cms.Task
    items += [('subtask', x) for x in getattr(thing, '_collection')]
  if thing: # for everything, esp. leaves like EDAnalyzer etc.
    pass

  for name, child in items:
    collect_trace(child, graph, entry)

def setupenv():
  bindir = tempfile.mkdtemp()
  print("+Setting up in ", bindir)
  os.symlink(__file__, bindir + "/cmsRun")
  os.environ["PATH"] = bindir + ":" + os.environ["PATH"]
  os.environ["CMSSWCALLTREE"] = bindir + "/" + OUTFILE_TREE
  with open(os.environ["CMSSWCALLTREE"], "w") as f:
    pass
  return bindir

def cleanupenv(tmpdir):
  #with open(os.environ["CMSSWCALLTREE"], "a") as f:
  #  print("}", file=f)
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

  # Search $PATH. There seems to be no pre-made function for this.
  for entry in path:
    file_path = os.path.join(entry, progname)
    if os.path.isfile(file_path):
      break
  if not os.path.isfile(file_path):
    print("+Cannot find program (%s) in modified $PATH (%s)." % (progname, path))
    sys.exit(1)
  print("+Found %s as %s in %s." % (progname, file_path, path))

  def writeoutput():
    print("+Done running %s, writing output..." % progname)

    def formatfile(filename):
      filename = os.path.abspath(filename)
      for pfx in STRIPPATHS():
        if filename.startswith(pfx):
          filename = filename[len(pfx):]
      return filename

    def format(event):
      evt, classname, filename, line = event
      filename = formatfile(filename)
      return "%s:%s; %s::%s" % (filename, line, classname, evt)
      
    with open(os.environ["CMSSWCALLTREE"], "a") as outfile:
        for child, parent in graph:
          print("%s -> %s" % (format(child), format(parent)), file=outfile)

  try:
    with open(file_path) as fp:
      code = compile(fp.read(), progname, 'exec')
      # try to emulate __main__ namespace as much as possible
      globals = {
      '__file__': progname,
      '__name__': '__main__',
      '__package__': None,
      '__cached__': None,
      }

      # now turn on the traceing
      instrument_cmssw()
      try:
        exec code in globals, globals
        # reporting is only possible if the config was executed successfully.
        import pickle
        collect_trace(globals["process"], graph, ('cmsRun', '', progname, 0))
        writeoutput()

      finally:
        sys.settrace(None)

  except OSError as err:
    print("+Cannot run file %r because: %s" % (sys.argv[0], err))
    sys.exit(1)
  except SystemExit:
    pass
  # this is not necessarily reached at all. 
  sys.exit(0)

def help():
  print("Usage: %s <some cmssw commandline>" % (sys.argv[0]))
  print("  The given programs will be executed, instrumenting calls cmsRun.")
  print("  cmsRun will not actually run cmssw, but all the Python code will be executed and instrumentd. The results are written to the file `%s` in the same directory." % OUTFILE_TREE)
  print("Examples:")
  print("  %s runTheMatrix.py -l 1000 --ibeos" % sys.argv[0])

def main():
  print("+Running cmsswfiletrace...")
  if sys.argv[0].endswith('cmsRun'):
      print("+Wrapping cmsRun...")
      trace_python(sys.argv[1:], ["."])
      return
  if len(sys.argv) <= 1:
    help()
    return
  # else
  print("+Running command with tracing %s..." % sys.argv[1:])
  trace_command(sys.argv[1:])

if __name__ == '__main__':
  main()
  
