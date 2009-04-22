import sys
import os
import logging
import random
import subprocess

from Vispa.Main.Exceptions import *
try:
    import FWCore.ParameterSet.SequenceTypes as sqt
    import FWCore.ParameterSet.Config as cms
    import FWCore.ParameterSet.Modules as mod
    import FWCore.ParameterSet.Types as typ
    import FWCore.ParameterSet.Mixins as mix
except Exception:
    logging.error(__name__ + ": " + exception_traceback())

from Vispa.Plugins.ConfigBrowser.ConfigDataAccessor import *
from Vispa.Plugins.ConfigBrowser.FileExportPlugin import *

class DotExport(FileExportPlugin):
  option_types={
    'legend':('Show Legend','boolean',True),
    'source':('Show Source','boolean',True),
    'es':('Show Event Setup','boolean',False),
    'tagconnect':('Connect InputTags','boolean',True),
    'seqconnect':('Connect Module Sequence','boolean',False),
    'services':('Show Services','boolean',False),
    'endpath':('Show EndPaths','boolean',True),
    'seq':('Group Sequences','boolean',False),
    'class':('Show Class','boolean',True),
    'file':('Show File','boolean',True)
  }
  plugin_name='DOT Export'
  file_types=('bmp','dot','eps','gif','jpg','pdf','png','ps','svg','tif')
  def __init__(self):
    FileExportPlugin.__init__(self)
    #self.options = {'legend':True,'source':True,'es':False,'tagconnect':False,'seqconnect':True,'services':True,'seq':True}
    
    self.shapes={}
    self.shapes['EDProducer']='box'
    self.shapes['EDFilter']='invhouse'
    self.shapes['EDAnalyzer']='house'
    self.shapes['OutputModule']='invtrapezium'
    self.shapes['ESSource']='Mdiamond'     
    self.shapes['ESProducer']='Msquare'
    self.shapes['Source']='ellipse'
    self.shapes['Service']='diamond'
  
  def produceDOT(self,data):
    
    def recurseChildren(obj):
      result=[]
      children=data.children(obj)
      if children:
        for c in children:
          result += recurseChildren(c)
      else:
        result.append(obj)
      return result
      
    def nodeLabel(obj):
      result = data.label(obj)
      if self.options['class']:
        result += '\\n%s'%data.classname(obj)
      if self.options['file']:
        result += '\\n%s:%s'%(data.filename(obj),data.lineNumber(obj))
      return result
    
    all={}
    for tlo in data.children(data.topLevelObjects()[0]):
      children = data.children(tlo)
      if children:
        all[tlo._label]=children
    
    nodes={}
    pathstarts=[]
    pathends=[]
    endstarts=[]
    maxpath=0
    maxendpath=0
    
    result='digraph configbrowse {\nsubgraph clusterProcess {\nlabel="%s"\n' % (data.process().name_())
      
    if 'paths' in all:
      for p in all['paths']:
        r = recurseChildren(p)
        result += 'subgraph cluster%s {\nlabel="%s"\ncolor=blue\n' % (data.label(p),data.label(p))
        if self.options['seqconnect']:
          pathstarts.append('start_%s'%data.label(p))
          pathends.append('end_%s'%data.label(p))
          nodes['start_%s'%data.label(p)]={'obj':p,'n_label':'Start %s'%data.label(p),'n_color':'grey','n_shape':'plaintext','inpath':False}
          nodes['end_%s'%data.label(p)]={'obj':p,'n_label':'End %s'%data.label(p),'n_color':'grey','n_shape':'plaintext','inpath':False}
        labels=[]
        for c in r:
          nodes[data.label(c)]={'obj':c,'n_label':nodeLabel(c),'n_shape':self.shapes.get(data.type(c),'plaintext'),'inpath':True}
          labels.append(data.label(c))
        if self.options['seqconnect']:
          result += '->'.join(labels)
        else:
          result += '\n'.join(labels)
        result += '\n}\n'
        if len(labels)>0 and self.options['seqconnect']:
          result += 'start_%s->%s\n%s->end_%s\n' % (data.label(p),labels[0],labels[-1],data.label(p))
        if len(labels)>maxpath:
          maxpath=len(labels)
    if self.options['endpath']:
      if 'endpaths' in all:
        for p in all['endpaths']:
          r = recurseChildren(p)
          result += 'subgraph cluster%s {\nlabel="%s"\ncolor=red\n' % (data.label(p),data.label(p))
          if self.options['seqconnect']:
            endstarts.append('endstart_%s'%data.label(p))
            nodes['endstart_%s'%data.label(p)]={'obj':p,'n_label':'Start %s'%data.label(p),'n_color':'grey','n_shape':'plaintext','inpath':False}
          labels=[]
          for c in r:
            nodes[data.label(c)]={'obj':c,'n_label':nodeLabel(c),'n_shape':self.shapes.get(data.type(c),'plaintext'),'inpath':True}
            labels.append(data.label(c))
          if self.options['seqconnect']:
            result += '->'.join(labels)
          else:
            result += '\n'.join(labels)
          result += '\n}\n'
          if len(labels)>0 and self.options['seqconnect']:
            result += 'endstart_%s->%s\n' % (data.label(p),labels[0])
          if len(labels)>maxendpath:
            maxendpath=len(labels)
    
    if self.options['seqconnect']:        
      for p in pathends:
        for p2 in endstarts:
          result+="%s->%s\n" % (p,p2)
    
    sequences={}
    if self.options['seq']:
      for n in [n for n in nodes if nodes[n]['inpath']]:
        foundin = data.foundIn(nodes[n]['obj'])
        for f in [f for f in foundin if data.type(f)=='Sequence']:
          if f in sequences:
            sequences[f].append(n)
          else:
            sequences[f]=[n]
      for s in sequences:
        result += 'subgraph clusterSequences%s {\nlabel="Sequence %s"\ncolor=green\n' % (s,s)
        for n in sequences[s]:
          result+='%s\n' % (n)
        result+= '}\n'
    
    
    if self.options['tagconnect']:
      allobjects = [nodes[n]['obj'] for n in nodes if nodes[n]['inpath']]
      data.readConnections(allobjects)
      connections = data.connections()
      for c in connections:
        result += '%s->%s[color=blue,label="%s",fontcolor=blue]' % (c[0],c[2],c[3])
      
    if self.options['source']:
      if 'source' in all:
        for s in all['source']:
          nodes['source']={'obj':s,'n_label':data.classname(s),'n_shape':self.shapes['Source']}
          if self.options['seqconnect']:
            for p in pathstarts:
              result += "source->%s\n" % (p)
        
    servicenodes=[]
            
    if self.options['es']:
      if 'essources' in all:
        for e in all['essources']:
          servicenodes.append(data.label(e))
          nodes[data.label(e)]={'obj':e,'n_label':nodeLabel(e), 'n_shape':self.shapes['ESSource'],'inpath':False}
      if 'esproducers' in all:
        for e in all['esproducers']:
          servicenodes.append(data.label(e))
          nodes[data.label(e)]={'obj':e,'n_label':nodeLabel(e), 'n_shape':self.shapes['ESProducer'],'inpath':False}
    if self.options['services']:
      if 'services' in all:
        for s in all['services']:
          servicenodes.append(data.label(s))
          nodes[data.label(s)]={'obj':s,'n_label':nodeLabel(e), 'n_shape':self.shapes['Service'],'inpath':False}
      
    for i,s in enumerate(servicenodes[:-1]):
      if not i%(maxpath+maxendpath)==(maxpath+maxendpath)-1:
        result+="%s->%s[style=invis]\n" % (s,servicenodes[i+1])
            
    for n in nodes:
      result += "%s[%s]\n" % (n,','.join(['%s="%s"' % (k[2:],v) for k,v in nodes[n].items() if k[0:2]=='n_']))
                
    
    result+='}\n'
    if self.options['legend']:
      result+=self.legend()
      
    result += "}\n"
    return result 

  def legend(self):
    return 'subgraph clusterLegend {\nlabel="legend"\ncolor=red\nSource->Producer->Filter->Analyzer\nService->ESSource[style=invis]\nESSource->ESProducer[style=invis]\nProducer->Filter[color=blue,label="InputTag",fontcolor=blue]\nProducer[shape=%s]\nFilter[shape=%s]\nAnalyzer[shape=%s]\nESSource[shape=%s]\nESProducer[shape=%s]\nSource[shape=%s]\nService[shape=%s]\n}\n' % (self.shapes['EDProducer'],self.shapes['EDFilter'],self.shapes['EDAnalyzer'],self.shapes['ESSource'],self.shapes['ESProducer'],self.shapes['Source'],self.shapes['Service'])
       
  def export(self,data,filename,filetype):
    if not data.process():
      raise "DOTExport requires a cms.Process object"  
    
    dot = self.produceDOT(data)
    
    if filetype=='dot':
      dotfile = open(filename,'w')
      dotfile.write(dot)
      dotfile.close()
    elif filetype=='stdout':
      print result
    elif filetype=='pdf':
      dot_p = subprocess.Popen(['dot','-Tps2'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
      ps2 = dot_p.communicate(dot)[0]
      pdf_p = subprocess.Popen(['ps2pdf','-',filename],stdin=subprocess.PIPE)
      pdf_p.communicate(ps2)
    else:
      subprocess.Popen(['dot','-T%s'%(filetype),'-o',filename],stdin=subprocess.PIPE).communicate(dot)
