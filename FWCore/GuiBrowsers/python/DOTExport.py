import sys
import os
import logging
import random
import subprocess
import re
import struct

import FWCore.ParameterSet.SequenceTypes as sqt
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Modules as mod
import FWCore.ParameterSet.Types as typ
import FWCore.ParameterSet.Mixins as mix

from Vispa.Plugins.ConfigEditor.ConfigDataAccessor import ConfigDataAccessor
from FWCore.GuiBrowsers.FileExportPlugin import FileExportPlugin

class DotProducer(object):
  def __init__(self,data,options,shapes):
    self.data = data
    self.options = options
    self.shapes = shapes
    self.nodes={}
    #lists of starts, ends of paths for path-endpath and source-path connections
    self.pathstarts=[]
    self.pathends=[]
    self.endstarts=[]
    self.toplevel = self.getTopLevel()
    
  def getTopLevel(self):
      
    #build a dictionary of available top-level objects
    all_toplevel={}
    if self.data.process():
      for tlo in self.data.children(self.data.topLevelObjects()[0]):
        children = self.data.children(tlo)
        if children:
          all_toplevel[tlo._label]=children
    else:
      #case if we have only an anonymous (non-process) file
      #pick up (modules, sequences, paths)
      for tlo in self.data.topLevelObjects():
        if self.data.type(tlo)=='Sequence':
          if 'sequences' in all_toplevel:
            all_toplevel['sequences']+=[tlo]
          else:
            all_toplevel['sequences']=[tlo]
        if self.data.type(tlo)=='Path':
          if 'paths' in all_toplevel:
            all_toplevel['paths']+=[tlo]
          else:
            all_toplevel['paths']=[tlo]
        if self.data.type(tlo) in ('EDAnalyzer','EDFilter','EDProducer','OutputModule'):
          self.nodes[self.data.label(tlo)]={'obj':tlo,'n_label':self.nodeLabel(tlo),'n_shape':self.shapes.get(self.data.type(tlo),'plaintext'),'inpath':True} 
        if self.options['services'] and self.data.type(tlo)=='Service':
          self.nodes[self.data.label(tlo)]={'obj':tlo,'n_label':self.nodeLabel(tlo),'n_shape':self.shapes.get(self.data.type(tlo),'plaintext'),'inpath':False}
        if self.options['es'] and self.data.type(tlo) in ('ESSource','ESProducer'):
          self.nodes[self.data.label(tlo)]={'obj':tlo,'n_label':self.nodeLabel(tlo),'n_shape':self.shapes.get(self.data.type(tlo),'plaintext'),'inpath':False}
    return all_toplevel      

  def seqRecurseChildren(self,obj):
    children = self.data.children(obj)
    if children:
      seqlabel = self.data.label(obj)
      if self.options['file']:
        seqlabel += '\\n%s:%s' % (self.data.pypackage(obj),self.data.lineNumber(obj))
      result='subgraph clusterSeq%s {\nlabel="Sequence %s"\ncolor="%s"\nfontcolor="%s"\nfontname="%s"\nfontsize=%s\n' % (self.data.label(obj),seqlabel,self.options['color_sequence'],self.options['color_sequence'],self.options['font_name'],self.options['font_size'])
      for c in children:
        result += self.seqRecurseChildren(c)
      result+='}\n'
      return result
    else:
      self.nodes[self.data.label(obj)]={'obj':obj,'n_label':self.nodeLabel(obj),'n_shape':self.shapes.get(self.data.type(obj),'plaintext'),'inpath':True}
      return '%s\n'%self.data.label(obj)   
      
  def recurseChildren(self,obj):
    result=[]
    children=self.data.children(obj)
    if children:
      for c in children:
        result += self.recurseChildren(c)
    else:
      result.append(obj)
    return result
    
  #write out an appropriate node label
  def nodeLabel(self,obj):
    result = self.data.label(obj)
    if self.options['class']:
      result += '\\n%s'%self.data.classname(obj)
    if self.options['file']:
      result += '\\n%s:%s'%(self.data.pypackage(obj),self.data.lineNumber(obj))
    return result
    
    #generate an appropriate URL by replacing placeholders in baseurl
  def nodeURL(self,obj):
    classname = self.data.classname(obj)
    pypath = self.data.pypath(obj)
    pyline = self.data.lineNumber(obj)
    url = self.options['urlbase'].replace('$classname',classname).replace('$pypath',pypath).replace('$pyline',pyline)
    return url
    
  def makePath(self,path,endpath=False):
    children = self.recurseChildren(path)
    pathlabel = self.data.label(path)
    if self.options['file']:
      pathlabel += '\\n%s:%s'%(self.data.pypackage(path),self.data.lineNumber(path))
    if endpath:
      pathresult = 'subgraph cluster%s {\nlabel="%s"\ncolor="%s"\nfontcolor="%s"\nfontname="%s"\nfontsize=%s\n' % (self.data.label(path),pathlabel,self.options['color_endpath'],self.options['color_endpath'],self.options['font_name'],self.options['font_size'])
    else:
      pathresult = 'subgraph cluster%s {\nlabel="%s"\ncolor="%s"\nfontcolor="%s"\nfontname="%s"\nfontsize=%s\n' % (self.data.label(path),pathlabel,self.options['color_path'],self.options['color_path'],self.options['font_name'],self.options['font_size'])
    if self.options['seqconnect']:
      if endpath:
        self.endstarts.append('endstart_%s'%self.data.label(path))
        self.nodes['endstart_%s'%self.data.label(path)]={'obj':path,'n_label':'Start %s'%self.data.label(path),'n_color':'grey','n_shape':'plaintext','inpath':False}
      else:
        self.pathstarts.append('start_%s'%self.data.label(path))
        self.pathends.append('end_%s'%self.data.label(path))
        self.nodes['start_%s'%self.data.label(path)]={'obj':path,'n_label':'Start %s'%self.data.label(path),'n_color':'grey','n_shape':'plaintext','inpath':False}
        self.nodes['end_%s'%self.data.label(path)]={'obj':path,'n_label':'End %s'%self.data.label(path),'n_color':'grey','n_shape':'plaintext','inpath':False}
    labels=[]
    for c in children:
      #this is also done in seqRecurseChildren, so will be duplicated
      #unncessary, but relatively cheap and saves more cff/cfg conditionals
      self.nodes[self.data.label(c)]={'obj':c,'n_label':self.nodeLabel(c),'n_shape':self.shapes.get(self.data.type(c),'plaintext'),'inpath':True}
      labels.append(self.data.label(c))
    if self.options['seqconnect']:
      pathresult += '->'.join(labels)+'\n'
    else:
      if not self.options['seq']:
        pathresult += '\n'.join(labels)+'\n'
    if self.options['seq']:
      if self.data.children(path):
        for path_child in self.data.children(path):
          pathresult += self.seqRecurseChildren(path_child)
    pathresult += '}\n'
    if len(labels)>0 and self.options['seqconnect']:
      if endpath:
        pathresult += 'endstart_%s->%s\n' % (self.data.label(path),labels[0])
      else:
        pathresult += 'start_%s->%s\n%s->end_%s\n' % (self.data.label(path),labels[0],labels[-1],self.data.label(path))
    
    return pathresult

  def producePaths(self):
    result=''
    if 'paths' in self.toplevel:
      for path in self.toplevel['paths']:
        result += self.makePath(path)
    if self.options['endpath']:
      if 'endpaths' in self.toplevel:
        for path in self.toplevel['endpaths']:
          result += self.makePath(path,True)
    if 'sequences' in self.toplevel:
      for seq in self.toplevel['sequences']:
        result += self.seqRecurseChildren(seq)
    return result
    
  def connectPaths(self):
    result=''
    for p in self.pathends:
      for p2 in self.endstarts:
        result+="%s->%s\n" % (p,p2)
    return result
  
  def connectTags(self):
    #if we are connecting by tag, add labelled tag joining lines
    #this doesn't have to be exclusive with sequence connection, by stylistically probably should be
    result=''
    allobjects = [self.nodes[n]['obj'] for n in self.nodes if self.nodes[n]['inpath']]
    self.data.readConnections(allobjects)
    connections = self.data.connections()
    for objects,names in connections.items():
      if self.options['taglabel']:
        result += '%s->%s[label="%s",color="%s",fontcolor="%s",fontsize=%s,fontname="%s"]\n' % (objects[0],objects[1],names[1],self.options['color_inputtag'],self.options['color_inputtag'],self.options['font_size'],self.options['font_name'])
      else:
        result += '%s->%s[color="%s"]\n' % (objects[0],objects[1],self.options['color_inputtag'])
    return result
  
  
  def produceSource(self):
    #add the source
    #if we are connecting sequences, connect it to all the path starts
    #if we are connecting sequences and have a schedule, connect it to path #0
    result=''
    if 'source' in self.toplevel:
      for s in self.toplevel['source']:
        self.nodes['source']={'obj':s,'n_label':self.data.classname(s),'n_shape':self.shapes['Source']}
        if self.options['seqconnect']:
            for p in self.pathstarts:
              result += 'source->%s\n' % (p)   
    return result
    
  def produceServices(self):
    # add service, eventsetup nodes
    # this will usually result in thousands and isn't that interesting
    servicenodes=[]
    result=''
    if self.options['es']:
      if 'essources' in self.toplevel:
        for e in self.toplevel['essources']:
          servicenodes.append(self.data.label(e))
          self.nodes[self.data.label(e)]={'obj':e,'n_label':self.nodeLabel(e), 'n_shape':self.shapes['ESSource'],'inpath':False}
      if 'esproducers' in self.toplevel:
        for e in self.toplevel['esproducers']:
          servicenodes.append(self.data.label(e))
          self.nodes[self.data.label(e)]={'obj':e,'n_label':self.nodeLabel(e), 'n_shape':self.shapes['ESProducer'],'inpath':False}
    if self.options['services']:
      if 'services' in self.toplevel:
        for s in self.toplevel['services']:
          self.servicenodes.append(self.data.label(s))
          self.nodes[self.data.label(s)]={'obj':s,'n_label':self.nodeLabel(e), 'n_shape':self.shapes['Service'],'inpath':False}
    #find the maximum path and endpath lengths for servicenode layout
    maxpath=max([len(recurseChildren(path) for path in self.toplevel.get('paths',(0,)))])
    maxendpath=max([len(recurseChildren(path) for path in self.toplevel.get('endpaths',(0,)))])
    
    #add invisible links between service nodes where necessary to ensure they only fill to the same height as the longest path+endpath
    #this constraint should only apply for link view
    for i,s in enumerate(servicenodes[:-1]):
      if not i%(maxpath+maxendpath)==(maxpath+maxendpath)-1:
        result+='%s->%s[style=invis]\n' % (s,servicenodes[i+1])
    return result
    
  def produceNodes(self):
    result=''
    for n in self.nodes:
      self.nodes[n]['n_fontname']=self.options['font_name']
      self.nodes[n]['n_fontsize']=self.options['font_size']
      if self.options['url']:
        self.nodes[n]['n_URL']=self.nodeURL(self.nodes[n]['obj'])
      result += "%s[%s]\n" % (n,','.join(['%s="%s"' % (k[2:],v) for k,v in self.nodes[n].items() if k[0:2]=='n_']))
    return result
    
  def produceLegend(self):
    """
    Return a legend subgraph using current shape and colour preferences.
    """
    return 'subgraph clusterLegend {\nlabel="legend"\ncolor=red\nSource->Producer->Filter->Analyzer\nService->ESSource[style=invis]\nESSource->ESProducer[style=invis]\nProducer->Filter[color="%s",label="InputTag",fontcolor="%s"]\nProducer[shape=%s]\nFilter[shape=%s]\nAnalyzer[shape=%s]\nESSource[shape=%s]\nESProducer[shape=%s]\nSource[shape=%s]\nService[shape=%s]\nsubgraph clusterLegendSequence {\nlabel="Sequence"\ncolor="%s"\nfontcolor="%s"\nProducer\nFilter\n}\n}\n' % (self.options['color_inputtag'],self.options['color_inputtag'],self.shapes['EDProducer'],self.shapes['EDFilter'],self.shapes['EDAnalyzer'],self.shapes['ESSource'],self.shapes['ESProducer'],self.shapes['Source'],self.shapes['Service'],self.options['color_sequence'],self.options['color_sequence'])
    
  def __call__(self):
    blocks=[]
    if self.options['legend']:
      blocks += [self.produceLegend()]
    blocks += [self.producePaths()]
    if self.options['seqconnect']:
      blocks += [self.connectPaths()]
    if self.options['tagconnect']:
      blocks += [self.connectTags()]
    if self.options['source']:
      blocks += [self.produceSource()]
    if self.options['es'] or self.options['services']:
      blocks += [self.produceServices()]
    blocks += [self.produceNodes()]
    if self.data.process():
      return 'digraph configbrowse {\nsubgraph clusterProcess {\nlabel="%s\\n%s"\nfontsize=%s\nfontname="%s"\n%s\n}\n}\n' % (self.data.process().name_(),self.data._filename,self.options['font_size'],self.options['font_name'],'\n'.join(blocks))
    else:
      return 'digraph configbrowse {\nsubgraph clusterCFF {\nlabel="%s"\nfontsize=%s\nfontname="%s"\n%s\n}\n}\n' % (self.data._filename,self.options['font_size'],self.options['font_name'],'\n'.join(blocks))
  
  

class DotExport(FileExportPlugin):
  """
  Export a CMSSW config file to DOT (http://www.graphviz.org) markup, either as raw markup or by invoking the dot program, as an image.
  """
  option_types={
    'legend':('Show Legend','boolean',True),
    'source':('Show Source','boolean',True),
    'es':('Show Event Setup','boolean',False),
    'tagconnect':('Connect InputTags','boolean',True),
    'seqconnect':('Connect Module Sequence','boolean',False),
    'services':('Show Services','boolean',False),
    'endpath':('Show EndPaths','boolean',True),
    'seq':('Group Sequences','boolean',True),
    'class':('Show Class','boolean',True),
    'file':('Show File','boolean',True),
    'schedule':('Show Schedule','boolean',False),
    'taglabel':('Show Tag Labels','boolean',True),
    'color_path':('Path Color','color','#ff00ff'),
    'color_endpath':('EndPath Color','color','#ff0000'),
    'color_sequence':('Sequence Color','color','#00ff00'),
    'color_inputtag':('InputTag Color','color','#0000ff'),
    'color_schedule':('Schedule Color','color','#00ffff'),
    'url':('Include URLs','boolean',False), #this is only purposeful for png+map mode
    'urlprocess':('Postprocess URL (for client-side imagemaps)','boolean',False), #see processMap documentation; determines whether to treat 'urlbase' as a dictionary for building a more complex imagemap or a simple URL
    'urlbase':('URL to generate','string',"{'split_x':1,'split_y':2,'scale_x':1.,'scale_y':1.,'cells':[{'top':0,'left':0,'width':1,'height':1,'html_href':'http://cmslxr.fnal.gov/lxr/ident/?i=$classname','html_alt':'LXR','html_class':'LXR'},{'top':1,'left':0,'width':1,'height':1,'html_href':'http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/$pypath?view=markup#$pyline','html_alt':'CVS','html_class':'CVS'}]}"), #CVS markup view doesn't allow line number links, only annotate view (which doesn't then highlight the code...)
    'node_graphs':('Produce individual graphs focussing on each node','boolean',False),
    'node_graphs_restrict':('Select which nodes to make node graphs for','string',''),
    'node_depth':('Search depth for individual node graphs','int',1),
    'font_name':('Font name','string','Times-Roman'),
    'font_size':('Font size','int',8),
    'png_max_size':('Maximum edge for png image','int',16768)
  }
  plugin_name='DOT Export'
  file_types=('bmp','dot','eps','gif','jpg','pdf','png','ps','svg','tif','png+map','stdout')
  def __init__(self):
    FileExportPlugin.__init__(self)
    
    #could make these customizeable in the general options dict
    self.shapes={}
    self.shapes['EDProducer']='box'
    self.shapes['EDFilter']='invhouse'
    self.shapes['EDAnalyzer']='house'
    self.shapes['OutputModule']='invtrapezium'
    self.shapes['ESSource']='Mdiamond'     
    self.shapes['ESProducer']='Msquare'
    self.shapes['Source']='ellipse'
    self.shapes['Service']='diamond'
    
  def dotIndenter(self,dot):
    """
    Simple indenter for dot output, mainly to prettify it for human reading.
    """
    spaces = lambda d: ''.join([space]*d)
    newdot = ""
    depth = 0
    space = '  '
    for line in dot.splitlines():
      if '{' in line:
        newdot += spaces(depth)+line+'\n'
        depth += 1
      elif '}' in line:
        depth -= 1
        newdot += spaces(depth)+line+'\n'
      else:
        newdot += spaces(depth)+line+'\n'
    return newdot
  
  def selectNode(self,dotdata,node,depth_s):
    depth = int(depth_s)
    backtrace=False
    if depth<0:
      depth = abs(depth)
      backtrace=True
    re_link = re.compile(r'^\s*?(\w*?)->(\w*?)(?:\[.*?\])?$',re.MULTILINE)
    re_nodedef = re.compile(r'^\s*?(\w*?)(?:\[.*?\])?$',re.MULTILINE)
    re_title = re.compile(r'^label=\"(.*?)\"$',re.MULTILINE)
    re_nodeprops = re.compile(r'^\s*?('+node+r')\[(.*?)\]$',re.MULTILINE)
    
    nodes = re_nodedef.findall(dotdata)
    if not node in nodes:
      raise Exception, "Selected node (%s) not found" % (node)
    links_l = re_link.findall(dotdata)
    links = {}
    for link in links_l:
      if not backtrace:
        if link[0] in links:
          links[link[0]] += [link[1]]
        else:
          links[link[0]] = [link[1]]
      if link[1] in links:
          links[link[1]] += [link[0]]
      else:
        links[link[1]] = [link[0]]
      
    def node_recursor(links,depthleft,start):
      if start in links:
        if depthleft==0:
          return links[start]+[start]
        else:
          result = [start]
          for l in links[start]:
            result.extend(node_recursor(links,depthleft-1,l))
          return result
      else:
        return [start]
    
    
    include_nodes = set(node_recursor(links,depth-1,node))
    include_nodes.add(node)
    
    class link_replacer:
      def __init__(self,include_nodes):
        self.include_nodes=include_nodes
      def __call__(self,match):
        if match.group(1) in self.include_nodes and match.group(2) in self.include_nodes:
          return match.group(0)
        return ''
    class node_replacer:
      def __init__(self,include_nodes):
        self.include_nodes=include_nodes
      def __call__(self,match):
        if match.group(1) in self.include_nodes:
          return match.group(0)
        return ''
    
    dotdata = re_link.sub(link_replacer(include_nodes),dotdata)
    dotdata = re_nodedef.sub(node_replacer(include_nodes),dotdata)
    dotdata = re_title.sub(r'label="\g<1>\\nDepth '+str(depth_s)+r' from node ' +node+r'"',dotdata,1)
    dotdata = re_nodeprops.sub('\\g<1>[\\g<2>,color="red"]',dotdata,1)
    
    return dotdata
   
  def processMap(self,mapdata):
    """
    Re-process the client-side image-map produces when png+map is selected.
    DOT will only ever put a single URL in the imagemap corresponding to a node, with the 'url' parameter (after html encoding) as the url, and the 'title' parameter as the title. This isn't useful behaviour for our purposes. We want probably several link areas, or a javascript link to make a menu appear, or other more complex behaviour.
    
    If the option 'urlprocess' is turned on, this function is called, and it expects to find a dictionary it can eval in the url parameter. I can't think of a less messy way of passing data to this function without having inner access to DOT at the moment.
    
    This function iterates through all the areas in the mapfile, replacing each one with one or more areas according to the rules in the dictionary stored in the URL parameter.
    
    The dictionary should have structure:
    {
      split_x:#,
      split_y:#,
      scale_x:#,
      scale_y:#,
      cells:[
              {
                top:#,
                left:#,
                width:#,
                height:#,
                html_attribute1:"...",
                html_attribute2:"..."
            ]
    }
    The imagemap is first scaled in size by scale_x and scale_y.
    It is then split into split_x*split_y rectangular cells.
    New areas are created for each defined cell with the defined top,left location and width,height. This will not check you aren't making new areas that overlap if you define them as such.
    The areas then get attributes defined by html_attribute fields - eg, 'html_href':'mypage.htm' becomes 'href'='mypage.htm' in the area. Probably you want as a minimum to define html_href and html_alt. It would also be useful to set html_class to allow highlighting of different link types, or html_onclick/onmouseover for more exotic behaviour.
    
    This will probably be quite sensitive to the presence of special characters, complex splitting schemes, etc. Use with caution.
    
    This may be somewhat replaceable with the <html_label> and cut-down table format that graphviz provides, but I haven't had much of an experiment with that.
    """
    new_areas=[]
    area_default = {'split_x':1,'scale_x':1.,'split_y':1,'scale_y':1.,'cells':[]}
    cell_default = {'top':0,'left':0,'width':1,'height':1,'html_href':'#'}
    re_area = re.compile('<area.*?/>',re.DOTALL)
    #sometimes DOT comes up with negative coordinates, so we need to deal with them here (so all the other links will work at least)
    re_content = re.compile('href="(.*?)" title=".*?" alt="" coords="(-?[0-9]{1,6}),(-?[0-9]{1,6}),(-?[0-9]{1,6}),(-?[0-9]{1,6})"',re.DOTALL)
    re_htmlunquote = re.compile('&#([0-9]{1,3});')
    mapdata = re_htmlunquote.sub(lambda x: chr(int(x.group(1))),mapdata)
    areas = re_area.findall(mapdata)
    for area in areas:
      #print area
      data = re_content.search(area)
      baseurl = data.group(1)
      x1,y1,x2,y2 = map(int,(data.group(2),data.group(3),data.group(4),data.group(5)))
      rad_x,rad_y = int((x2-x1)*0.5),int((y2-y1)*0.5)
      centre_x,centre_y = x1+rad_x,y1+rad_y
      basedict = eval(baseurl)
      for ad in area_default:
        if not ad in basedict:
          basedict[ad]=area_default[ad]
      rad_x = int(rad_x*basedict['scale_x'])
      rad_y = int(rad_y*basedict['scale_y'])
      top_x,top_y = centre_x-rad_x,centre_y-rad_y
      split_x,split_y = int((2*rad_x)/basedict['split_x']),int((2*rad_y)/basedict['split_y'])
      
      for cell in basedict['cells']:
        for cd in cell_default:
          if not cd in cell:
            cell[cd]=cell_default[cd]
        x1,y1 = top_x+split_x*cell['left'],top_y+split_y*cell['top']
        x2,y2 = x1+split_x*cell['width'],y1+split_y*cell['height']
        area_html = '<area shape="rect" coords="%s,%s,%s,%s" %s />' % (x1,y1,x2,y2,' '.join(['%s="%s"'%(key.split('_',1)[1],value) for key, value in cell.items() if key.startswith('html_')]))
        new_areas.append(area_html)
    return '<map id="configbrowse" name="configbrowse">\n%s\n</map>'%('\n'.join(new_areas))  
    
    
  def export(self,data,filename,filetype):
    #if not data.process():
    #  raise "DOTExport requires a cms.Process object"  
    
    #dot = self.produceDOT(data)
    dot_producer = DotProducer(data,self.options,self.shapes)
    dot = dot_producer()
    
    if len(dot_producer.nodes)>0:
    
      if self.options['node_graphs']:
        nodes = [n for n in dot_producer.nodes if data.type(dot_producer.nodes[n]['obj']) in ('EDAnalyzer','EDFilter','EDProducer','OutputModule')]
        for n in nodes:
          if self.options['node_graphs_restrict'] in n:
            try:
              node_dot = self.selectNode(dot,n,self.options['node_depth'])
              self.write_output(node_dot,filename.replace('.','_%s.'%n),filetype)
            except:
              pass
      else:
        dot = self.dotIndenter(dot)
        self.write_output(dot,filename,filetype)
    else:
      print "WARNING: Empty image. Not saved."
    
  
  def get_png_size(self,filename):
    png_header = '\x89PNG\x0d\x0a\x1a\x0a'
    ihdr = 'IHDR'
    filedata = open(filename,'r').read(24)
    png_data = struct.unpack('>8s4s4sII',filedata)
    if not (png_data[0]==png_header and png_data[2]==ihdr):
      raise 'PNG header or IHDR not found'
    return png_data[3],png_data[4]
    
  def write_output(self,dot,filename,filetype):
    #don't use try-except-finally here, we want any errors passed on so the enclosing program can decide how to handle them
    if filetype=='dot':
      dotfile = open(filename,'w')
      dotfile.write(dot)
      dotfile.close()
    elif filetype=='stdout':
      print result
    elif filetype=='pdf':
      dot_p = subprocess.Popen(['dot','-Tps2'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
      ps2 = dot_p.communicate(dot)[0]
      if not dot_p.returncode==0:
        raise "dot returned non-zero exit code: %s"%dot_p.returncode
      pdf_p = subprocess.Popen(['ps2pdf','-',filename],stdin=subprocess.PIPE)
      pdf_p.communicate(ps2)
      if not pdf_p.returncode==0:
        raise "ps2pdf returned non-zero exit code: %s"%pdf_p.returncode
    elif filetype=='png+map':
      if '.' in filename:
        filename = filename.split('.')[0]
      dot_p = subprocess.Popen(['dot','-Tpng','-o','%s.png'%filename,'-Tcmapx_np'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
      mapdata = dot_p.communicate(dot)[0]
      if not dot_p.returncode==0:
        raise "dot returned non-zero exit code: %s"%dot_p.returncode
      if self.options['urlprocess']:
        mapdata = self.processMap(mapdata)
      mapfile = open('%s.map'%filename,'w')
      mapfile.write(mapdata)
      mapfile.close()
      filesize = self.get_png_size('%s.png'%filename)
      if max(filesize) > self.options['png_max_size']:
        print "png image is too large (%s pixels/%s max pixels), deleting" % (filesize,self.options['png_max_size'])
        os.remove('%s.png'%filename)
        os.remove('%s.map'%filename)
    elif filetype=='png':
      dot_p = subprocess.Popen(['dot','-T%s'%(filetype),'-o',filename],stdin=subprocess.PIPE)
      dot_p.communicate(dot)
      if not dot_p.returncode==0:
        raise "dot returned non-zero exit code: %s"%dot_p.returncode
      filesize = self.get_png_size(filename)
      if max(filesize) > self.options['png_max_size']:
        print "png image is too large (%s pixels/%s max pixels), deleting" % (filesize,self.options['png_max_size'])
        os.remove(filename)
    else:
      dot_p = subprocess.Popen(['dot','-T%s'%(filetype),'-o',filename],stdin=subprocess.PIPE)
      dot_p.communicate(dot)
      if not dot_p.returncode==0:
        raise "dot returned non-zero exit code: %s"%dot_p.returncode
    
    

