#!/usr/bin/env python

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet import DictTypes

import sys, os, os.path

# enable tracing cms.Sequences, cms.Paths and cms.EndPaths for all imported modules (thus, process.load(...), too)
import tracingImport
import six

result = dict()
result['procname']       = ''
result['main_input']     = None
result['looper']         = DictTypes.SortedKeysDict()
result['psets']          = DictTypes.SortedKeysDict()
result['modules']        = DictTypes.SortedKeysDict()
result['es_modules']     = DictTypes.SortedKeysDict()
result['es_sources']     = DictTypes.SortedKeysDict()
result['es_prefers']     = DictTypes.SortedKeysDict()
result['output_modules'] = list()
result['sequences']      = DictTypes.SortedKeysDict()
result['paths']          = DictTypes.SortedKeysDict()
result['endpaths']       = DictTypes.SortedKeysDict()
result['services']       = DictTypes.SortedKeysDict()
result['schedule']       = ''

def dumpObject(obj,key):
    if key in ('es_modules','es_sources','es_prefers'):
        classname = obj['@classname']
        label = obj['@label']
        del obj['@label']
        del obj['@classname']
        returnString = "{'@classname': %s, '@label': %s, %s" %(classname, label, str(obj).lstrip('{'))
        return returnString
    elif key in ('modules','services'):
        classname = obj['@classname']
        del obj['@label']
        del obj['@classname']
        returnString = "{'@classname': %s, %s" %(classname, str(obj).lstrip('{'))
        return returnString
    elif key in ('psets',):
        returnString = "('PSet', 'untracked', %s)" % str(obj)
        return returnString
    else:
        return str(obj)


def trackedness(item):
  if item.isTracked():
    return 'tracked'
  else:
    return 'untracked'

# the problem are non empty VPsets
def fixup(item):
  if isinstance(item, bool):
    if item: return 'true'
    else: return 'false'
  elif isinstance(item, list):
      return [str(i) for i in item]
  elif isinstance(item, str):
      return '"%s"' %item
  else:
      return str(item)

def prepareParameter(parameter):
    if isinstance(parameter, cms.VPSet):
        configValue = []
        for item in parameter:
            configValue.append((prepareParameter(item)[2]))
        return (type(parameter).__name__, trackedness(parameter), configValue )
    if isinstance(parameter, cms.PSet):
        configValue = {}
        for name, item in six.iteritems(parameter.parameters_()):
          configValue[name] = prepareParameter(item)
        return (type(parameter).__name__, trackedness(parameter), configValue )
    else:
        return (type(parameter).__name__, trackedness(parameter), fixup(parameter.value()) )

def parsePSet(module):
  if module is None: return
  config = DictTypes.SortedKeysDict()
  for parameterName,parameter in six.iteritems(module.parameters_()):
    config[parameterName] = prepareParameter(parameter)
  return config

def parseSource(module):
  if module is None: return
  config = DictTypes.SortedKeysDict()
  config['@classname'] = ('string','tracked',module.type_())
  for parameterName,parameter in six.iteritems(module.parameters_()):
    config[parameterName] = prepareParameter(parameter)
  return config

def parseModule(name, module):
  if module is None: return
  config = DictTypes.SortedKeysDict()
  config['@classname'] = ('string','tracked',module.type_())
  config['@label'] = ('string','tracked',name)
  for parameterName,parameter in six.iteritems(module.parameters_()):
    config[parameterName] = prepareParameter(parameter)
  return config

def parseModules(process):
  result['procname'] = process.process
 
  result['main_input'] = parseSource(process.source)

  for name,item in six.iteritems(process.producers):
    result['modules'][name] = parseModule(name, item)

  for name,item in six.iteritems(process.filters):
    result['modules'][name] = parseModule(name, item)

  for name,item in six.iteritems(process.analyzers):
    result['modules'][name] = parseModule(name, item)

  for name,item in six.iteritems(process.outputModules):
    result['modules'][name] = parseModule(name, item)
    result['output_modules'].append(name)

  for name,item in six.iteritems(process.es_sources):
    result['es_sources'][name + '@'] = parseModule(name, item)

  for name,item in six.iteritems(process.es_producers):
    result['es_modules'][name + '@'] = parseModule(name, item)

  for name,item in six.iteritems(process.es_prefers):
    result['es_prefers'][name + '@'] = parseModule(name, item)

  for name,item in six.iteritems(process.psets):
    result['psets'][name] = parsePSet(item)

  for name,item in six.iteritems(process.sequences):
    result['sequences'][name] = "'" + item.dumpConfig("")[1:-2] + "'"

  for name,item in six.iteritems(process.paths):
    result['paths'][name] = "'" + item.dumpConfig("")[1:-2] + "'"

  for name,item in six.iteritems(process.endpaths):
    result['endpaths'][name] = "'" + item.dumpConfig("")[1:-2] + "'"

  for name,item in six.iteritems(process.services):
    result['services'][name] = parseModule(name, item)

  # TODO still missing:
  #   process.vpsets
  #   process.looper
  #   process.schedule
 
  # use the ordering seen at module import time for sequence, paths and endpaths,
  # keeping only those effectively present in the process
  # (some might have been commented, removed, not added, whatever...)
  result['paths'].list     = [ path for path in tracingImport.original_paths     if path in result['paths'].list ]
  result['endpaths'].list  = [ path for path in tracingImport.original_endpaths  if path in result['endpaths'].list ]
  result['sequences'].list = [ path for path in tracingImport.original_sequences if path in result['sequences'].list ]

  # nothing to do for 'main_input' as it's a single item
  
  # sort alphabetically everything else
  result['modules'].list.sort()
  result['output_modules'].sort()
  result['es_sources'].list.sort()
  result['es_modules'].list.sort()
  result['es_prefers'].list.sort()
  result['psets'].list.sort()
  result['services'].list.sort()


# find and load the input file
sys.path.append(os.environ["PWD"])
filename = sys.argv[1].rstrip('.py')
theConfig = __import__(filename)

try: 
    #'process' in theConfig.__dict__:
    theProcess = theConfig.process
except:
    # what if the file is just a fragment ?
    # try to load it into a brand new process...
    theProcess = cms.Process('')
    try:
        theProcess.load(filename)
    except:
        sys.err.write('Unable to parse configuation fragment %s into a new Process\n' % sys.argv[1])
        sys.exit(1)

# parse the configuration
parseModules(theProcess)

# now dump it to the screen as wanted by the HLT parser
hltAcceptedOrder = ['main_input','looper', 'psets', 'modules', 'es_modules', 'es_sources', 'es_prefers', 'output_modules', 'sequences', 'paths', 'endpaths', 'services', 'schedule']

print '{'
print "'procname': '%s'" %result['procname']

for key in hltAcceptedOrder:
    if key in ('output_modules', 'schedule'):
        print ", '%s': %s" %(key, result[key])
    elif key in ('main_input',):
        print ", '%s':  {" % key
        # in case no source is defined, leave an empty block in the output
        if result[key] is not None:
            print str(dumpObject(result[key], key))[1:-1]
        print '} # end of %s' % key
    else:
        print ", '%s':  {" % key
        comma = ''
        for name,object in six.iteritems(result[key]):
            print comma+"'%s': %s" %(name, dumpObject(object,key))
            comma = ', '
        print '} # end of %s' % key

print '}'

