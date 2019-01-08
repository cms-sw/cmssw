#!/usr/bin/env python2
from __future__ import print_function
import os
import sys
import trace
import tempfile
import subprocess
from shutil import copy, rmtree

# only needed to locate CMSSW
import FWCore

WRAP_SCRIPTS = ["cmsDriver.py" ]
IGNORE_DIRS = [
  os.path.abspath(sys.prefix),
  os.path.dirname(FWCore.__file__),
]


def setupenv():
  bindir = tempfile.mkdtemp()
  for s in WRAP_SCRIPTS:
    os.symlink(__file__, bindir + "/" + s)
  os.symlink(__file__, bindir + "/cmsRun")
  os.environ["PATH"] = bindir + ":" + os.environ["PATH"]
  os.environ["CMSSWCALLTREE"] = bindir + "/calltree"
  with open(os.environ["CMSSWCALLTREE"], "w") as f:
    print("digraph cmsswpythoncalltree {", file=f)
  return bindir

def cleanupenv(tmpdir):
  with open(os.environ["CMSSWCALLTREE"], "a") as f:
    print("}", file=f)
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
  t = trace.Trace(count=0, trace=0, countfuncs=0, countcallers=1, ignoredirs=IGNORE_DIRS)
  sys.argv = prog_argv
  progname = prog_argv[0]
  sys.path[0] = os.path.split(progname)[0]

  # Search $PATH. There seems to be no pre-made function for this.
  for entry in path:
    file_path = os.path.join(entry, progname)
    if os.path.isfile(file_path):
      break
  print("Found %s as %s in %s." % (progname, file_path, path))
  if not os.path.isfile(file_path):
    print("Cannot find program (%s) in modified $PATH (%s)." % (progname, path))
    sys.exit(1)

  try:
    with open(file_path) as fp:
      code = compile(fp.read(), progname, 'exec')
      # try to emulate __main__ namespace as much as possible
      globs = {
      '__file__': progname,
      '__name__': '__main__',
      '__package__': None,
      '__cached__': None,
      }
      t.runctx(code, globs, globs)
  except OSError as err:
    print("Cannot run file %r because: %s" % (sys.argv[0], err))
    sys.exit(1)
  except SystemExit:
    pass

  results = t.results()

  with open(os.environ["CMSSWCALLTREE"], "a") as outfile:
    for ((pfile, pmod, pfunc), (cfile, cmod, cfunc)) in sorted(results.callers):
      print('  "%s.%s" -> "%s.%s" [label="%s"]' % (pmod, pfunc, cmod, cfunc, progname), file=outfile)

def main():
  for s in WRAP_SCRIPTS:
    if sys.argv[0].endswith(s):
      tmppath = os.path.dirname(sys.argv[0])
      path = filter(
        lambda s: not s.startswith(tmppath),
        os.environ["PATH"].split(":")
      )
      trace_python([s] + sys.argv[1:], path)
      return
  if sys.argv[0].endswith('cmsRun'):
      trace_python(sys.argv[1:], ["."])
      return

  if len(sys.argv) <= 1:
    print("Usage: %s <some cmssw commandline>" % sys.argv[0])
  else:
    trace_command(sys.argv[1:])


if __name__ == '__main__':
  main()
  
