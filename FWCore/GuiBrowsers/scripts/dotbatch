#! /usr/bin/env python

from optparse import OptionParser
import Vispa.Plugins.ConfigBrowser.ConfigDataAccessor as cda
import Vispa.Plugins.ConfigBrowser.DOTExport as de
import os.path

"""
Command line tool for batch generation of dot images.

Probably you want to use it like 'python dotbatch.py --preset tag /path/to/config/files/*.py'

The two preset mode generate either module-order (link) or input-tag-order (tag) graphs.
"""

presets = {
  'tag':{'endpath':False,'source':False,'legend':False},
  'link':{'seqconnect':True,'tagconnect':False,'legend':False}
}

parser = OptionParser(usage='Usage: %prog [options] file0_cfg.py file1_cfg.py)')

parser.add_option('-f','--format',dest='filetype',help='Output format to use (eg png, pdf, dot).',type='string',default='png',metavar='FORMAT')
parser.add_option('-n','--name',dest='naming',help='Name to use for output files. Defaults config_cfg.(format).',type='string',default='$o.$f',metavar='NAME')
parser.add_option('-o','--opts',dest='opts',help='Options for DotExport.',type='string',metavar='opt1:val1;opt2:val2',default=None)
parser.add_option('-p','--preset',dest='preset',help='Use a preset mode.\nKnown modes: %s'%presets.keys(),metavar='PRESET',default=None,type='string')
parser.add_option('-v','--verbose',dest='verbose',help='Verbose mode',default=False,action='store_true')
parser.add_option('-s','--stop',dest='stop',help='Stop on errors',default=False,action='store_true')

(options,args)=parser.parse_args()

dot = de.DotExport()

filetype = options.filetype
naming = options.naming

if options.verbose:
  print 'dotbatch starting...'

if options.opts:
  for o in options.opts.split(';'):
    if ':' in o:
      opt,val = o.split(':')
      if options.verbose:
        print 'Setting option %s=%s' % (opt,val)
      dot.setOption(opt,val)

if options.preset:
  if options.preset in presets:
    if options.verbose:
      print 'Setting preset mode: %s'%options.preset
    for opt,val in presets[options.preset].items():
      dot.setOption(opt,val)
  else:
    print 'Preset mode: %s not found'%options.preset
      
for file in args:
  filename = os.path.basename(file)
  if filename.endswith('.py'):
    filename=filename[:-3]
  output = naming.replace('$o',filename).replace('$f',filetype)
  if options.verbose:
    print 'Processing file: %s -> %s' % (file,output),
  try:
    data = cda.ConfigDataAccessor()
    data.open(file)
    dot.export(data,output,filetype)
  except:
    if options.stop:
      raise
  if options.verbose:
    print '...done'

if options.verbose:
  print 'dotbatch finished' 
