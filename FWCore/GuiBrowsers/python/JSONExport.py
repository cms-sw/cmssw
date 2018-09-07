import sys
import os.path
import logging
import random

import FWCore.ParameterSet.SequenceTypes as sqt
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Modules as mod
import FWCore.ParameterSet.Types as typ
import FWCore.ParameterSet.Mixins as mix

from Vispa.Plugins.ConfigEditor.ConfigDataAccessor import ConfigDataAccessor
from FWCore.GuiBrowsers.FileExportPlugin import FileExportPlugin

class JsonExport(FileExportPlugin):
  option_types={}
  plugin_name='JSON Export'
  file_types=('html','json')
  def __init__(self):
    FileExportPlugin.__init__(self)

  def produce(self,data):
    
    #pset = lambda pdict: [[k,repr(v).split('(',1)[0],(repr(v).split('(',1)[1][:-1])] for k,v in pdict.items()]
    def pset(pdict):
      result = []
      for k,v in pdict.items():
        if v.pythonTypeName()=='cms.PSet' or v.pythonTypeName()=='cms.untracked.PSet':
          result.append([k,v.pythonTypeName(),'pset',pset(v.parameters_())])
        elif v.pythonTypeName()=='cms.VPSet' or v.pythonTypeName()=='cms.untracked.VPSet':
          result.append([k,v.pythonTypeName(),'vpset',[pset(a.parameters_()) for a in v]])
        elif v.pythonTypeName().lower().startswith('cms.v') or v.pythonTypeName().lower().startswith('cms.untracked.v'):
          result.append([k,v.pythonTypeName(),'list',[repr(a) for a in v]])
        else:
          result.append([k,v.pythonTypeName(),'single',repr(v.pythonValue())])
      return result
          
    #allObjects = [d for d in data._allObjects if (data.type(d) in ("EDProducer","EDFilter","EDAnalyzer","OutputModule"))]
    #data.readConnections(allObjects)
        
    def moduledict(mod,prefix,links=False):
      result={}
      result['label']=data.label(mod)
      result['class']=data.classname(mod)
      result['file']=data.pypath(mod)
      result['line']=data.lineNumber(mod)
      result['package']=data.pypackage(mod)
      result['pset']=pset(mod.parameters_())
      result['type']=data.type(mod)
      if links:
        result['uses']=[data.uses(mod)]
        result['usedby']=[data.usedBy(mod)]
      result['id']='%s_%s'%(prefix,data.label(mod))
      return result
      
    all={}
    for tlo in data.topLevelObjects():
      children=data.children(tlo)
      if children:
        all[tlo._label]=children
      
    process = {'name':data.process().name_(),'src':data._filename}
    
    #now unavailable  
    #schedule = []
    #if 'Schedule' in all:
    #  for s in all['Schedule']:
    #    schedule.append(data.label(s))
      
    source={}
    if 'source' in all:
      s = all['source'][0]
      source['class']=data.classname(s)
      source['pset']=pset(s.parameters_())
      
    essources=[]
    if 'essources' in all:
      for e in all['essources']:
        essources.append(moduledict(e,'essource'))
    esproducers=[]
    if 'esproducers' in all:
      for e in all['esproducers']:
        essources.append(moduledict(e,'esproducer'))
    esprefers=[]
    if 'esprefers' in all:
      for e in all['esprefers']:
        essources.append(moduledict(e,'esprefers'))
    services=[]
    if 'services' in all:
      for s in all['services']:
        services.append({'class':data.classname(s),'pset':pset(s.parameters_())})    
      
      
    def jsonPathRecursive(p,prefix):
      #print "At:",self.label(p),self.type(p)
      children = data.children(p)
      if children:
        children = [jsonPathRecursive(c,prefix) for c in children]
        return {'type':'Sequence','label':'Sequence %s'%(data.label(p)),'id':'seq_%s' % data.label(p),'children':children}
      else:
        return moduledict(p,prefix,True)
        
          
    paths=[]
    if 'paths' in all:
      for p in all['paths']:
        path=jsonPathRecursive(p,data.label(p))
        if path:
          if not isinstance(path, type([])):
            if path['type']=='Sequence':
              path = path['children']
            else:
              path = [path]
        else:
          path=[]
        paths.append({'label':data.label(p),'path':path})
    endpaths=[]
    if 'endpaths' in all:
      for p in all['endpaths']:
        path=jsonPathRecursive(p,data.label(p))
        if path:
          if not isinstance(path, type([])):
            if path['type']=='Sequence':
              path = path['children']
            else:
              path = [path]
        else:
          path=[]
        endpaths.append({'label':data.label(p),'path':path})
      
    #json={'process':process,'schedule':schedule,'source':source,'essources':essources,'esproducers':esproducers,'esprefers':esprefers,'services':services,'paths':paths,'endpaths':endpaths}
    json={'process':process,'source':source,'essources':essources,'esproducers':esproducers,'esprefers':esprefers,'services':services,'paths':paths,'endpaths':endpaths}
      
    return repr(json)
    
  def export(self,data,filename,filetype):
    if not data.process():
      raise "JSONExport requires a cms.Process object"
      
    json = self.produce(data)
    
    if filetype=='json':
      jsonfile = open(filename,'w')
      jsonfile.write(json)
      jsonfile.close()
    if filetype=='html':
      #open the HTML template and inject the JSON...
      pass
      
