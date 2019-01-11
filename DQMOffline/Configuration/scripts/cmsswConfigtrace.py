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

import cmsswFiletrace

cmsswFiletrace.OUTFILE_TREE = "configtree"
cmsswFiletrace.OUTFILE_FILES = "configfiles"
cmsswFiletrace.WRAP_SCRIPTS = []
IGNORE_PACKAGES = ['FWCore/ParameterSet', 'FWCore/GuiBrowsers', 'DQMOffline/Configuration/scripts', "cmsRun"]

from cmsswFiletrace import *

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

def trace_location(thing, name, extra = lambda thing, *args, **kwargs: thing):
  old_method = getattr(thing, name)
  def trace_location_hook(self, *args, **kwargs):
    retval = old_method(self, *args, **kwargs)
    where = auto_inspect()
    #print("Called %s::%s at %s" % (thing.__name__, name, where[0][1:3]))
    event = (name, where[0][1:3], extra(self, *args, **kwargs))
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


def instrument_cmssw():
  # everything happens in the imports so far
  pass

def collect_trace(thing, name, graph, parent):
  # thing is what to look at, graph is the output list (of child, parent tuple pairs)
  # thing could be pretty much anything.
  classname = thing.__class__.__name__
  if hasattr(thing, '_trace_events'):
    events = getattr(thing, '_trace_events')
    for action, loc, extra in events:
      filename = loc[0]
      line = loc[1]
      entry = (action, classname, filename, line)
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
    print("No _trace_events found in %s.\nMaybe turn on tracing for %s?" % (thing, classname))
    print(" Found in %s" % (parent,))


def writeoutput(graph):
  progname = ", ".join(PREFIXINFO)
  print("+Done running %s, writing output..." % progname)

  def formatfile(filename):
    filename = os.path.abspath(filename)
    for pfx in STRIPPATHS:
      if filename.startswith(pfx):
        filename = filename[len(pfx):]
    return filename

  def format(event):
    evt, classname, filename, line = event
    filename = formatfile(filename)
    return "%s:%s; %s::%s" % (filename, line, classname, evt)

  files = set()
  for child, parent, relation in graph:
    files.add(child[2])
    files.add(parent[2])
  with open(os.environ["CMSSWCALLFILES"], "a") as outfile:
    for f in files:
      print("%s: %s" % (progname, formatfile(f)), file=outfile)
    
  with open(os.environ["CMSSWCALLTREE"], "a") as outfile:
      for child, parent, relation in graph:
        print("%s -> %s [%s]" % (format(child), format(parent), relation), file=outfile)

def trace_python(prog_argv, path):
  graph = []

  sys.argv = prog_argv
  progname = prog_argv[0]

  file_path = searchinpath(progname, path)
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
        collect_trace(globals["process"], 'cmsrun', graph, ('cmsRun', '', progname, 0))
        writeoutput(graph)

      finally:
        sys.settrace(None)

  except OSError as err:
    print("+Cannot run file %r because: %s" % (sys.argv[0], err))
    sys.exit(1)
  except SystemExit:
    pass
  # this is not necessarily reached at all. 
  sys.exit(0)
cmsswFiletrace.trace_python = trace_python

def help():
  print("Usage: %s <some cmssw commandline>" % (sys.argv[0]))
  print("  The given programs will be executed, instrumenting calls to cmsRun.")
  print("  cmsRun will not actually run cmssw, but all the Python code will be executed and instrumentd. The results are written to the file `%s` in the same directory." % OUTFILE_TREE)
  print("  The callgraph output lists edges pointing from each function to the one calling it.")
  print("Examples:")
  print("  %s runTheMatrix.py -l 1000 --ibeos" % sys.argv[0])
cmsswFiletrace.help = help

if __name__ == '__main__':
  main()
  
