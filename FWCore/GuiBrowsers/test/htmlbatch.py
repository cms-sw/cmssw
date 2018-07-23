#!/usr/bin/env python

from __future__ import print_function
import os
import sys
from optparse import OptionParser

try:
    distBaseDirectory=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    if not os.path.exists(distBaseDirectory) or not "Vispa" in os.listdir(distBaseDirectory):
        distBaseDirectory=os.path.abspath(os.path.join(os.path.dirname(__file__),"../python"))
    if not os.path.exists(distBaseDirectory) or not "Vispa" in os.listdir(distBaseDirectory):
        distBaseDirectory=os.path.abspath(os.path.expandvars("$CMSSW_BASE/python/FWCore/GuiBrowsers"))
    if not os.path.exists(distBaseDirectory) or not "Vispa" in os.listdir(distBaseDirectory):
        distBaseDirectory=os.path.abspath(os.path.expandvars("$CMSSW_RELEASE_BASE/python/FWCore/GuiBrowsers"))
except Exception:
    distBaseDirectory=os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]),".."))

sys.path.append(distBaseDirectory)

from Vispa.Plugins.ConfigEditor.ConfigDataAccessor import ConfigDataAccessor
from FWCore.GuiBrowsers.HTMLExport import HTMLExport
import os

def verbose(msg):
  if options.verbose:
    print(msg)

parser = OptionParser(usage="Usage: %prog [options] file0_cfg.py file1_cfg.py...")

parser.add_option('-s','--stop',dest='stop',help='Stop on errors',default=False,action='store_true')
parser.add_option('-v','--verbose',dest='verbose',help='Verbose mode',default=False,action='store_true')

(options,args)=parser.parse_args()

html_export = HTMLExport()

verbose('htmlbatch starting...')

for file in args:
  if not os.path.exists(file):
    verbose('File does not exist: %s'%file)
    if options.stop:
      break
    else:
      continue
  filename = os.path.basename(file)
  if filename.endswith('.py'):
    filename=filename[:-3]
  output = filename+'.html'
  verbose('Processing file: %s -> %s' % (file,output))
  try:
    verbose('\tLoading data')
    data = ConfigDataAccessor()
    data.open(file)
    verbose('\tExporting')
    html_export.export(data,output,'html')
    verbose('\tDone')
  except:
    verbose('\tERROR')
    if options.stop:
      raise
    else:
      continue

verbose('htmlbatch finished')
  
