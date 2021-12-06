#!/usr/bin/env python3

from __future__ import print_function
from contentValuesLib import *

class OptionParser(optparse.OptionParser):
  """ Option parser class """
  def __init__(self):
    optparse.OptionParser.__init__(self, usage="%prog [options] root_file ...", version="%prog 0.0.1", conflict_handler="resolve")
    self.add_option("--silent", "-s", action="store_true", dest="silent", default=False, help="silent mode: specify return code and exit")
    self.add_option("--subsystem", "-c", action="store", type="string", dest="subsystem", default=None, help="Specify test subsystem")

if __name__ == "__main__":
  
  # Create option parser and get options/arguments
  optManager  = OptionParser()
  (opts, args) = optManager.parse_args()
  opts = opts.__dict__

  # Check if at least one root file defined (can be many!)
  if len(args) == 0:
    print("At least one ROOT file must be priovided, use --help for hit")
    sys.exit(1)

  # Check if all files exists and are accessible
  for rfile in args:
    try:
      os.stat(rfile)
    except:
      print("File [", rfile, "] not exists or is not accessible?")
      sys.exit(2)

  ss = opts['subsystem']

  # Lets extract values from files one-by-one, construct hashmap and check values
  for rfile in args:

    (run_number, values) = getSummaryValues(file_name = rfile, shift_type = None, translate = False, filters = None)

    if values == None or len(values) == 0:
      print("No content summary values found. Skipping file: %s" % rfile)
      continue

    messages = []
    for sub in SUBSYSTEMS.keys():

      if not ss == None and not sub == ss:
        continue

      if sub not in values:
        messages.append("%s: missing subsystem!" % sub)
        continue

      skeys = {}
      sfolders = []

      for folder in FOLDERS.keys():

        if folder not in values[sub]:
          messages.append("%s: missing folder EventInfo/%s" % (sub, folder))
          continue

        if len(values[sub][folder]) == 0:
          messages.append("%s: empty folder EventInfo/%s" % (sub, FOLDERS[folder][1]))
          continue

        sfolders.append(folder)

        if 'Summary' not in values[sub][folder]:
          messages.append("%s: missing summary value EventInfo/%s" % (sub, FOLDERS[folder][1]))

        for key in values[sub][folder].keys():
          if key == 'Summary':
            continue
          if key not in skeys:
            skeys[key] = []
          skeys[key].append(folder)
          
      for key in skeys:
        nfound = []
        for folder in sfolders:
          if skeys[key].count(folder) == 0: nfound.append(folder)
        if len(nfound) > 0:
          messages.append("%s: value (%s)/%s not found in (%s)" % (sub, ','.join(skeys[key]), key, ','.join(nfound)))

    if not opts['silent']:
      for message in sorted(messages):  print(message)          
      print("%d errors found" % len(messages))

    if len(messages) > 0: sys.exit(1)

  sys.exit(0)


