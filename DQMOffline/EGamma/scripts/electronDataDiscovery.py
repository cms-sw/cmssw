#!/usr/bin/env python

import os, sys
if len(sys.argv) > 1:
  os.environ['DD_SOURCE'] = sys.argv[1]

import DQMOffline.EGamma.electronDataDiscovery as dd

if os.environ['DD_TIER_SECONDARY'] == "":
  files = dd.search()
  print "dataset has", len(files), "files:"
  for file in files:
    print file
else:
  files = dd.search()
  print "dataset has", len(files), "primary files:"
  for file in files:
    print file
  files = dd.search2()
  print "dataset has", len(files), "secondary files:"
  for file in files:
    print file

	
	

