#!/usr/bin/env python2.4
import FWCore.ParameterSet.Config as cms



import sys, os
sys.path.append(os.environ["PWD"])

filename = sys.argv[1].rstrip('.py')

theConfig = __import__(filename)
theConfig = theConfig.process
result = {}

result["procname"] = ''
result['main_input'] = {}
result['looper'] = {}
result['psets'] = {}
result['modules'] = {}
result['es_modules'] = {}
result['es_sources'] = {}
result['es_prefers'] = {}
result['output_modules'] = []
result['sequences'] = {}
result['paths'] = {}
result['endpaths'] = {}
result['services'] = {}
result['schedule'] = ''


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
        returnString = "('PSet', 'untracked', %s)"%str(obj)
        return returnString
    else:
        return obj


def trackedness(item):
  if item.isTracked:
    return 'tracked'
  else:
    return 'untracked'

#the problem are non empty VPsets
def fixup(item):
  if type(item) == bool:
    if item: return 'true'
    else: return 'false'  
  elif type(item) == list:
      return [str(i) for i in item]
  elif type(item) == str:
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
        for name, item in parameter.parameters_().iteritems():
          configValue[name] = prepareParameter(item)
        return (type(parameter).__name__, trackedness(parameter), configValue )
    else:      
        return ( type(parameter).__name__, trackedness(parameter), fixup(parameter.value()) )


hasSeenSource = False 
#loop through the file 
for name,item in theConfig.__dict__.iteritems():
  config = {}

  if isinstance(item,(cms.EDFilter,cms.EDProducer,cms.EDAnalyzer,cms.OutputModule)):
    config['@classname'] = ('string','tracked',item.type_())
    config["@label"] = ('string','tracked',name)

    for parameterName,parameter in item.parameters_().iteritems():
      config[parameterName] = prepareParameter(parameter)
    result['modules'][name] = config

    if isinstance(item,cms.OutputModule):
       result['output_modules'].append(name)

  elif isinstance(item,cms.Source):
      if not hasSeenSource:
         config['@classname'] = ('string','tracked',item.type_())
         for parameterName,parameter in item.parameters_().iteritems():
             config[parameterName] = prepareParameter(parameter)
         result['main_input'][name] = config
         hasSeenSource = True
      else: pass   

  elif isinstance(item,cms.ESProducer):
    config['@classname'] = ('string','tracked',item.type_())
    config["@label"] = ('string','tracked',name)
    name = name+'@'
    for parameterName,parameter in item.parameters_().iteritems():
        config[parameterName] = prepareParameter(parameter)
    result['es_modules'][name] = config

  elif isinstance(item,cms.ESSource):            
    config['@classname'] = ('string','tracked',item.type_())
    config["@label"] = ('string','tracked',name)
    name = name+'@'
    for parameterName,parameter in item.parameters_().iteritems():
        config[parameterName] = prepareParameter(parameter)
    result['es_sources'][name] = config

  elif isinstance(item,cms.ESPrefer):
    config['@classname'] = ('string','tracked',item.type_())
    config["@label"] = ('string','tracked',name)
    name = name+'@'
    for parameterName,parameter in item.parameters_().iteritems():
        config[parameterName] = prepareParameter(parameter)
    result['es_prefers'][name] = config

  elif isinstance(item, cms.PSet):
    for parameterName,parameter in item.parameters_().iteritems():
      config[parameterName] = prepareParameter(parameter)
    result['psets'][name] = config

  elif isinstance(item,cms.Sequence):
    result['sequences'][name] = "'"+item.dumpConfig("")[1:-2]+"'"

  elif isinstance(item,cms.EndPath):
      result['endpaths'][name] = "'"+item.dumpConfig("")[1:-2]+"'"

  elif isinstance(item,cms.Path):
      result['paths'][name] = "'"+item.dumpConfig("")[1:-2]+"'"

  elif isinstance(item,cms.Service):
      config['@classname'] = ('string','tracked',item.type_())
      config["@label"] = ('string','tracked',name)

      for parameterName,parameter in item.parameters_().iteritems():
          config[parameterName] = prepareParameter(parameter)
      result['services'][name] = config
                            

# now dump it to the screen
# as wanted by the HLT parser

hltAcceptedOrder = ['main_input','looper',  'psets',  'modules',  'es_modules',  'es_sources', 'es_prefers',  'output_modules',  'sequences',  'paths',  'endpaths',  'services',  'schedule']

print '{'
print "'procname': '%s'" %result['procname']


for key in hltAcceptedOrder:
    if key in ('output_modules', 'schedule'):
        print ", '%s': %s" %(key, result[key])
    elif key in ('main_input',):
        print ", '%s':  {" %key
        comma = ''
        for name,object in result[key].iteritems():
            print comma+str(dumpObject(object,key))[1:-1]
            comma = ', '
        print '} # end of %s' %key
    else:
        print ", '%s':  {" %key
        comma = ''
        for name,object in result[key].iteritems():
            print comma+"'%s': %s" %(name, dumpObject(object,key)) 
            comma = ', ' 
        print '} # end of %s' %key


print '}'  

