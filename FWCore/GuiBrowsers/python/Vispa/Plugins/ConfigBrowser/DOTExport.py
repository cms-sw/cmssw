import sys
import os
import logging
import random
import subprocess
import re

from Vispa.Main.Exceptions import *
try:
    import FWCore.ParameterSet.SequenceTypes as sqt
    import FWCore.ParameterSet.Config as cms
    import FWCore.ParameterSet.Modules as mod
    import FWCore.ParameterSet.Types as typ
    import FWCore.ParameterSet.Mixins as mix
except Exception:
    raise ImportError("cannot import CMSSW: " + exception_traceback())

from Vispa.Plugins.ConfigBrowser.ConfigDataAccessor import *
from Vispa.Plugins.ConfigBrowser.FileExportPlugin import *

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
    'urlbase':('URL to generate','string',"{'split_x':1,'split_y':2,'scale_x':1.,'scale_y':1.,'cells':[{'top':0,'left':0,'width':1,'height':1,'html_href':'http://cmslxr.fnal.gov/lxr/ident/?i=$classname','html_alt':'LXR','html_class':'LXR'},{'top':1,'left':0,'width':1,'height':1,'html_href':'http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/$pythonpath?view=markup','html_alt':'CVS','html_class':'CVS'}]}") # valid $classname, $pythonfile, $pythonpackage, $pythonpath, $pythonline
  }
  plugin_name='DOT Export'
  file_types=('bmp','dot','eps','gif','jpg','pdf','png','ps','svg','tif','png+map')
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
  
  def produceDOT(self,data):
    """
    Produce a string of DOT representing the supplied ConfigDataAccessor's currently loaded file, according to the current object options.
    """  
  
  
    #produce nested markup of children and sequences
    def seqRecurseChildren(obj):
      children = data.children(obj)
      if children:
        seqlabel = data.label(obj)
        if self.options['file']:
          seqlabel += '\\n%s:%s' % (data.filename(obj),data.lineNumber(obj))
        result='subgraph clusterSeq%s {\nlabel="Sequence %s"\ncolor="%s"\nfontcolor="%s"\n' % (data.label(obj),seqlabel,self.options['color_sequence'],self.options['color_sequence'])
        for c in children:
          result += seqRecurseChildren(c)
        result+='}\n'
        return result
      else:
        return '%s\n'%data.label(obj)
    
    #get a list of all an object's children, recursively
    def recurseChildren(obj):
      result=[]
      children=data.children(obj)
      if children:
        for c in children:
          result += recurseChildren(c)
      else:
        result.append(obj)
      return result
      
    #write out an appropriate node label
    def nodeLabel(obj):
      result = data.label(obj)
      if self.options['class']:
        result += '\\n%s'%data.classname(obj)
      if self.options['file']:
        result += '\\n%s:%s'%(data.filename(obj),data.lineNumber(obj))
      return result
    
    #generate an appropriate URL by replacing placeholders in baseurl
    def nodeURL(obj):
      classname = data.classname(obj)
      pythonfile = data.filename(obj)
      pythonpath = data.fullFilename(obj)
      pythonpackage = data.package(obj)
      pythonline = data.lineNumber(obj)
      url = self.options['urlbase'].replace('$classname',classname).replace('$pythonfile',pythonfile).replace('$pythonpath',pythonpath).replace('$pythonpackage',pythonpackage).replace('$pythonline',pythonline)
      return url
       
    def makePath(path,endpath=False):
      children = recurseChildren(path)
      pathlabel = data.label(path)
      if self.options['file']:
        pathlabel += '\\n%s:%s'%(data.filename(path),data.lineNumber(path))
      if endpath:
        pathresult = 'subgraph cluster%s {\nlabel="%s"\ncolor="%s"\nfontcolor="%s"\n' % (data.label(path),pathlabel,self.options['color_endpath'],self.options['color_endpath'])
      else:
        pathresult = 'subgraph cluster%s {\nlabel="%s"\ncolor="%s"\nfontcolor="%s"\n' % (data.label(path),pathlabel,self.options['color_path'],self.options['color_path'])
      if self.options['seqconnect']:
        if endpath:
          endstarts.append('endstart_%s'%data.label(path))
          nodes['endstart_%s'%data.label(path)]={'obj':path,'n_label':'Start %s'%data.label(path),'n_color':'grey','n_shape':'plaintext','inpath':False}
        else:
          pathstarts.append('start_%s'%data.label(path))
          pathends.append('end_%s'%data.label(path))
          nodes['start_%s'%data.label(path)]={'obj':path,'n_label':'Start %s'%data.label(path),'n_color':'grey','n_shape':'plaintext','inpath':False}
          nodes['end_%s'%data.label(path)]={'obj':path,'n_label':'End %s'%data.label(path),'n_color':'grey','n_shape':'plaintext','inpath':False}
      labels=[]
      for c in children:
        nodes[data.label(c)]={'obj':c,'n_label':nodeLabel(c),'n_shape':self.shapes.get(data.type(c),'plaintext'),'inpath':True}
        labels.append(data.label(c))
      if self.options['seqconnect']:
        pathresult += '->'.join(labels)+'\n'
      else:
        if not self.options['seq']:
          pathresult += '\n'.join(labels)+'\n'
      if self.options['seq']:
        if data.children(path):
          for path_child in data.children(path):
            pathresult += seqRecurseChildren(path_child)
      pathresult += '}\n'
      if len(labels)>0 and self.options['seqconnect']:
        if endpath:
          pathresult += 'endstart_%s->%s\n' % (data.label(path),labels[0])
        else:
          pathresult += 'start_%s->%s\n%s->end_%s\n' % (data.label(path),labels[0],labels[-1],data.label(path))
      
    
      return pathresult
        
      
    #build a dictionary of available top-level objects
    all_toplevel={}
    for tlo in data.topLevelObjects():
      children = data.children(tlo)
      if children:
        all_toplevel[tlo._label]=children
    
    #dictionary of all nodes that ultimately need to be added with style definitions
    #keys named n_foo are added as foo=value to the dot format
    nodes={}
    
    #lists of starts, ends of paths for path-endpath and source-path connections
    pathstarts=[]
    pathends=[]
    endstarts=[]
        
    #declare the toplevel graph
    result='digraph configbrowse {\nsubgraph clusterProcess {\nlabel="%s\\n%s"\n' % (data.process().name_(),data._filename)
    
    if 'Schedule(Paths)' in all_toplevel:
      for path in [path for path in all_toplevel['Schedule(Paths)'] if not (path in all_toplevel.get('EndPaths',()))]:
        result += makePath(path)
    if 'Paths' in all_toplevel:
      for path in [path for path in all_toplevel['Paths'] if not (path in all_toplevel.get('EndPaths',()))]:
        result += makePath(path)
    if self.options['endpath']:
      if 'EndPaths' in all_toplevel:
        for path in all_toplevel['EndPaths']:
          result += makePath(path,True)
    
    #if we are connecting by sequence, connect all path ends to all endpath starts
    if self.options['seqconnect']:
      if self.options['schedule']:
        if 'Schedule(Paths)' in all_toplevel:
          result += 'subgraph clusterSchedule {\nlabel="Schedule"\ncolor="%s"\nfontcolor="%s"\n' % (self.options['color_schedule'],self.options['color_schedule'])
          result += '->'.join(['start_%s' % data.label(path) for path in all_toplevel['Schedule(Paths)']])+'\n'
          result += '}\n'
      for p in pathends:
        for p2 in endstarts:
          result+="%s->%s\n" % (p,p2)
         
    
    #if we are connecting by tag, add labelled tag joining lines
    #this doesn't have to be exclusive with sequence connection, by stylistically probably should be
    if self.options['tagconnect']:
      allobjects = [nodes[n]['obj'] for n in nodes if nodes[n]['inpath']]
      data.readConnections(allobjects)
      connections = data.connections()
      for c in connections:
        if self.options['taglabel']:
          result += '%s->%s[label="%s",color="%s",fontcolor="%s"]\n' % (c[0],c[2],c[3],self.options['color_inputtag'],self.options['color_inputtag'])
        else:
          result += '%s->%s[color="%s"]\n' % (c[0],c[2],self.options['color_inputtag'])
    
    #add the source
    #if we are connecting sequences, connect it to all the path starts
    #if we are connecting sequences and have a schedule, connect it to path #0
    if self.options['source']:
      if 'Source' in all_toplevel:
        for s in all_toplevel['Source']:
          nodes['source']={'obj':s,'n_label':data.classname(s),'n_shape':self.shapes['Source']}
          if self.options['seqconnect']:
            if 'Schedule(Paths)' in all_toplevel and self.options['schedule']:
              if all_toplevel['Schedule(Paths)']:
                result += 'source->%s\n' % (data.label(data.children(all_toplevel['Schedule(Paths)'])[0]))
            else:
              for p in pathstarts:
                result += "source->%s\n" % (p)
        
    
    # add service, eventsetup nodes
    # this will usually result in thousands and isn't that interesting
    servicenodes=[]
    if self.options['es']:
      if 'ESSources' in all_toplevel:
        for e in all_toplevel['ESSources']:
          servicenodes.append(data.label(e))
          nodes[data.label(e)]={'obj':e,'n_label':nodeLabel(e), 'n_shape':self.shapes['ESSource'],'inpath':False}
      if 'ESProducers' in all_toplevel:
        for e in all_toplevel['ESProducers']:
          servicenodes.append(data.label(e))
          nodes[data.label(e)]={'obj':e,'n_label':nodeLabel(e), 'n_shape':self.shapes['ESProducer'],'inpath':False}
    if self.options['services']:
      if 'Services' in all_toplevel:
        for s in all_toplevel['Services']:
          servicenodes.append(data.label(s))
          nodes[data.label(s)]={'obj':s,'n_label':nodeLabel(e), 'n_shape':self.shapes['Service'],'inpath':False}
    
    #find the maximum path and endpath lengths for servicenode layout
    maxpath=max(max([len(recurseChildren(path)) for path in all_toplevel.get('Paths',(0,))]),max([len(recurseChildren(path)) for path in all_toplevel.get('Schedule(Paths)',(0,))]))
    maxendpath=max([len(recurseChildren(path)) for path in all_toplevel.get('EndPaths',(0,))])
    
    #add invisible links between service nodes where necessary to ensure they only fill to the same height as the longest path+endpath
    #this constraint should only apply for link view
    for i,s in enumerate(servicenodes[:-1]):
      if not i%(maxpath+maxendpath)==(maxpath+maxendpath)-1:
        result+="%s->%s[style=invis]\n" % (s,servicenodes[i+1])
            
    for n in nodes:
      if self.options['url']:
        nodes[n]['n_URL']=nodeURL(nodes[n]['obj'])
      result += "%s[%s]\n" % (n,','.join(['%s="%s"' % (k[2:],v) for k,v in nodes[n].items() if k[0:2]=='n_']))
                
    
    result += '}\n'
    if self.options['legend']:
      result+=self.legend()
      
    result += "}\n"
    return result 

  def legend(self):
    """
    Return a legend subgraph using current shape and colour preferences.
    """
    return 'subgraph clusterLegend {\nlabel="legend"\ncolor=red\nSource->Producer->Filter->Analyzer\nService->ESSource[style=invis]\nESSource->ESProducer[style=invis]\nProducer->Filter[color="%s",label="InputTag",fontcolor="%s"]\nProducer[shape=%s]\nFilter[shape=%s]\nAnalyzer[shape=%s]\nESSource[shape=%s]\nESProducer[shape=%s]\nSource[shape=%s]\nService[shape=%s]\nsubgraph clusterLegendSequence {\nlabel="Sequence"\ncolor="%s"\nfontcolor="%s"\nProducer\nFilter\n}\n}\n' % (self.options['color_inputtag'],self.options['color_inputtag'],self.shapes['EDProducer'],self.shapes['EDFilter'],self.shapes['EDAnalyzer'],self.shapes['ESSource'],self.shapes['ESProducer'],self.shapes['Source'],self.shapes['Service'],self.options['color_sequence'],self.options['color_sequence'])
  
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
    if not data.process():
      raise "DOTExport requires a cms.Process object"  
    
    dot = self.produceDOT(data)
    dot = self.dotIndenter(dot)
    
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
    else:
      dot_p = subprocess.Popen(['dot','-T%s'%(filetype),'-o',filename],stdin=subprocess.PIPE)
      dot_p.communicate(dot)
      if not dot_p.returncode==0:
        raise "dot returned non-zero exit code: %s"%dot_p.returncode
    

