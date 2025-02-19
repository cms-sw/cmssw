#!/usr/bin/env python

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

import Vispa.Plugins.ConfigEditor.ConfigDataAccessor as cda
import FWCore.GuiBrowsers.DOTExport as de
import os.path

"""
Command line tool for batch generation of dot images.

Probably you want to use it like 'python dotbatch.py --preset tag /path/to/config/files/*.py'

The two preset mode generate either module-order (link) or input-tag-order (tag) graphs.
"""

def verbose(msg):
  if options.verbose:
    print msg

presets = {
  'tag':{'endpath':False,'source':False,'legend':False},
  'link':{'seqconnect':True,'tagconnect':False,'legend':False}
}

parser = OptionParser(usage='Usage: %prog [options] file0_cfg.py file1_cfg.py)')

parser.add_option('-f','--format',dest='filetype',help='Output format to use (eg png, pdf, dot).',type='string',default='png',metavar='FORMAT')
parser.add_option('-n','--name',dest='naming',help='Name to use for output files. Defaults config_cfg.(format).',type='string',default='$o.$f',metavar='NAME')
parser.add_option('-o','--opts',dest='opts',help='Options for DotExport.',type='string',metavar='opt1:val1,opt2:val2',default=None)
parser.add_option('-p','--preset',dest='preset',help='Use a preset mode.\nKnown modes: %s'%presets.keys(),metavar='PRESET',default=None,type='string')
parser.add_option('-v','--verbose',dest='verbose',help='Verbose mode',default=False,action='store_true')
parser.add_option('-s','--stop',dest='stop',help='Stop on errors',default=False,action='store_true')

(options,args)=parser.parse_args()

dot = de.DotExport()

filetype = options.filetype
naming = options.naming

verbose('dotbatch starting...')

if options.preset:
  if options.preset in presets:
    verbose('Setting preset mode: %s'%options.preset)
    for opt,val in presets[options.preset].items():
      dot.setOption(opt,val)
  else:
    verbose('Preset mode: %s not found'%options.preset)

if options.opts:
  for o in options.opts.split(','):
    if ':' in o:
      opt,val = o.split(':')
      verbose('Setting option %s=%s' % (opt,val))
      dot.setOption(opt,eval(val))

for file in args:
  if not os.path.exists(file):
    verbose('File does not exist: %s'%file)
    continue
  filename = os.path.basename(file)
  if filename.endswith('.py'):
    filename=filename[:-3]
  output = naming.replace('$o',filename).replace('$f',filetype)
  verbose('Processing file: %s -> %s' % (file,output))
  try:
    verbose('    Loading data')
    data = cda.ConfigDataAccessor()
    data.open(file)
    verbose('    Exporting')
    dot.export(data,output,filetype)
    verbose('    Done')
  except:
    verbose('    ERROR')
    if options.stop:
      raise

verbose('dotbatch finished')
  

  
