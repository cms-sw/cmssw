#!/usr/bin/env python2
from __future__ import print_function
import os
import sys
import inspect
import tempfile
import subprocess
from shutil import copy, rmtree
from collections import defaultdict

# this already does a good job, but it is not enough
#import FWCore.GuiBrowsers.EnablePSetHistory

# we need to patch out this to allow printing unlabled things
import FWCore.ParameterSet.Mixins
del FWCore.ParameterSet.Mixins._Labelable.__str__

# then also trace Sequence construction so we can get a full tree
# (PSetHistory only tracks leaves)
IGNORE_PACKAGES = ['FWCore/ParameterSet', 'FWCore/GuiBrowsers', 'DQMOffline/Configuration/scripts']
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
      if issubclass(self.__class__, FWCore.ParameterSet.Mixins._Parameterizable):
        # there are some checks on some classes that need bypassing
        super(FWCore.ParameterSet.Mixins._Parameterizable,self). \
          __setattr__("_trace_events", [ event ])
      else:
        self.__setattr__("_trace_events", [ event ])

    return old_method(self, *args, **kwargs)
  setattr(thing, name, trace_location_hook)

from FWCore.ParameterSet.SequenceTypes import _ModuleSequenceType, _Sequenceable
from FWCore.ParameterSet.Modules import _Module
# with this we can also track the '+' and '*' of modules, but it is slow
#trace_location(_Sequenceable, '__init__')
trace_location(_Module, '__init__')
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

import FWCore
import FWCore.Modules.EmptySource_cfi

OUTFILE_TREE = "calltree"
def STRIPPATHS():
  # this can only be evaluated after FWCore is loaded, and instrumented
  return [
    "/".join(os.path.dirname(FWCore.__file__).split("/")[:-1]) + "/",
    "/".join(FWCore.Modules.EmptySource_cfi.__file__.split("/")[:-3]) + "/",
  ]

def instrument_cmssw():
  # everything happnes in the imports so far
  pass

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
  callgraph = defaultdict(lambda: set())


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
      
    def format(func):
      filename, funcname = func
      return "%s::%s" % (formatfile(filename), funcname)

    def callpath(func):
      # climb up in the call graph until we find a node without callers (this is
      # the entry point, the traced call itself). There may be cycles, but any
      # node is reachable from the entry point, so no backtracking required.
      path = []
      seen = set()
      parents = {func}
      while parents:
        if len(parents) == 1:
          func = next(iter(parents))
          seen.add(func)
          path.append(format(func))
        if len(parents) > 1:
          for func in parents:
            if not func in seen:
              break
          seen.add(func)
          path.append(format(func) + "+")
        parents = callgraph[func]
      return path[:-1]

    with open(os.environ["CMSSWCALLTREE"], "a") as outfile:
        for func in callgraph.keys():
          print("%s: %s 1" % (progname, ";".join(reversed(callpath(func)))), file=outfile)

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
        with open("config.dump", "wb") as f:
          pickle.dump(globals["process"], f)
        print(globals["process"])
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
  
