import os.path; global CONFIGDIR
from glob import glob

CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]

LAYOUTS = glob("%s/layouts/[^-_]*-layouts.py" % CONFIGDIR)
LAYOUTS += glob("%s/layouts/shift_[^-_]*_layout.py" % CONFIGDIR)
LAYOUTS += glob("%s/layouts/.*_overview_layouts.py" % CONFIGDIR)
