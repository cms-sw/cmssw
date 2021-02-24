import os.path; global CONFIGDIR
from glob import glob
import os.path, socket 

def reglob(pattern):
  """Extended version of glob that uses regular expressions."""
  from os import listdir
  import re
  cwd = pattern.rsplit('/',1)[0]
  f_pattern= pattern.rsplit('/',1)[-1]
  pat=re.compile(f_pattern)
  g = ["%s/%s" % (cwd,f) for f in listdir(cwd) if pat.match(f)]
  return g

CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]

LAYOUTS = reglob("%s/layouts/[^-_]*-layouts.py" % CONFIGDIR)
LAYOUTS += reglob("%s/layouts/shift_[^-_]*_layout.py" % CONFIGDIR)
LAYOUTS += reglob("%s/layouts/.*_overview_layouts.py" % CONFIGDIR)
