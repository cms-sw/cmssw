import sys
import os.path
import logging
import random

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

class HTMLExport(FileExportPlugin):
  options_types={}
  plugin_name='HTML Export'
  file_types=('html',)
  def __init__(self):
    FileExportPlugin.__init__(self)
  
  def produce(self,data):
    def elem(elemtype,innerHTML='',html_class='',**kwargs):
      if html_class:
        kwargs['class']=html_class
      return "<%s %s>%s</%s>\n" % (elemtype,' '.join(['%s="%s"'%(k,v) for k,v in kwargs.items()]),innerHTML,elemtype)
    def div(innerHTML='',html_class='',**kwargs):
      return elem('div',innerHTML,html_class,**kwargs)
    
    def htmlPSet(pset):
      def linkInputTag(tag):
        inputtag=''
        if isinstance(tag,typ.InputTag):
          inputtag = tag.pythonValue()
        else:
          inputtag = tag
        if len(str(tag))==0:
          inputtag = '""'
        return inputtag

      pset_items_html=''
      for k,v in pset.items():
        if isinstance(v,mix._ParameterTypeBase):
          if isinstance(v,mix._SimpleParameterTypeBase):
            item_class='other'
            if isinstance(v,typ.bool):
              item_class='bool'
            if isinstance(v,typ.double):
              item_class='double'
            if isinstance(v,typ.string):
              item_class='string'
            if isinstance(v,(typ.int32, typ.uint32, typ.int64, typ.uint64)):
              item_class='int'
            pset_items_html+=elem('tr',
              elem('td',k,'param-name')
             +elem('td',v.pythonTypeName(),'param-class')
             +elem('td',v.pythonValue(),'param-value-%s'%item_class),
             'pset-item'
            )
          if isinstance(v,typ.InputTag):
            pset_items_html+=elem('tr',
              elem('td',k,'param-name')
             +elem('td',v.pythonTypeName(),'param-class')
             +elem('td',linkInputTag(v),'param-value-inputtag'),
             'pset-item'
            )
          if isinstance(v,typ.PSet):
            pset_html = ''
            if len(v.parameters_())==0:
              pset_items_html+=elem('tr',
                elem('td',k,'param-name')
               +elem('td',v.pythonTypeName(),'param-class')
               +elem('td','(empty)','label'),
               'pset-item'
              )
            else:
              pset_items_html+=elem('tr',
                elem('td',k,'param-name')
               +elem('td',v.pythonTypeName(),'param-class')
               +elem('td',htmlPSet(v.parameters_())),
               'pset-item'
              )
          if isinstance(v,mix._ValidatingListBase):
            list_html = ''
            if len(v)==0:
              list_html = elem('li','(empty)','label')
            else:
              if isinstance(v,typ.VInputTag):
                for vv in v:
                  list_html += elem('li',linkInputTag(vv),'param-value-inputtag pset-list-item')
              elif isinstance(v,typ.VPSet):
                for vv in v:
                  list_html += elem('li',htmlPSet(vv.parameters_()),'pset-list-item')
              else:
                item_class='other'
                if isinstance(v,typ.vbool):
                  item_class='bool'
                if isinstance(v,typ.vdouble):
                  item_class='double'
                if isinstance(v,typ.vstring):
                  item_class='string'
                if isinstance(v,(typ.vint32,typ.vuint32,typ.vint64,typ.vuint64)):
                  item_class='int'
                for vv in v:
                  if len(str(vv))==0:
                    vv = "''"
                  list_html += elem('li',vv,'pset-list-item param-value-%s'%item_class)
            pset_items_html+=elem('tr',
              elem('td',k,'param-name')
             +elem('td',v.pythonTypeName(),'param-class')
             +elem('td',elem('ol',list_html,'pset-list')),
             'pset-item'
            )
              
            
      return elem('table',pset_items_html,'pset')
      
    def htmlModule(mod):
      mod_label_html = div(elem('a',data.label(mod),'title',name=data.label(mod)),'module_label '+data.type(mod),onClick='return toggleVisible(\'%s\')'%('mod_%s'%(data.label(mod))))
      
      mod_table = elem('table',
        elem('tr',elem('td','Type','label')+elem('td',data.type(mod)))
       +elem('tr',elem('td','Class','label')+elem('td',data.classname(mod))),
        'module_table')
        
      mod_pset = htmlPSet(mod.parameters_())
      
      mod_content_html = div(mod_table+mod_pset,'module_area',id='mod_%s'%data.label(mod))
      return div(mod_label_html+mod_content_html,'module')
      
    def htmlPathRecursive(p):
      children = data.children(p)
      if children:
        seq_name='Sequence'
        if isinstance(p,sqt.Path):
          seq_name='Path'
        if isinstance(p,sqt.EndPath):
          seq_name='EndPath'
        seq_label_html = div('%s %s'%(seq_name,elem('span',data.label(p),'title')),'sequence_label',onClick='return toggleVisible(\'%s\')'%'seq_%s'%data.label(p))
        seq_inner_content_html = ''.join([htmlPathRecursive(c) for c in children])
        seq_content_html = div(seq_inner_content_html,'sequence_area',id='seq_%s'%data.label(p))
        return div(seq_label_html+seq_content_html,'sequence')
      else:
        return htmlModule(p)
        
    toplevel={}
    for tlo in data.children(data.topLevelObjects()[0]):
      children = data.children(tlo)
      if children:
        toplevel[tlo._label]=children
        
    path_html=''
    if 'paths' in toplevel:
      for path in toplevel['paths']:
        path_html += div(htmlPathRecursive(path),'path')
    
    header_html = div('Config File Visualisation','header')
    file_html = div(elem('span','Process:','label')
                   +elem('span',data.process().name_(),'title')
                   +elem('span',data._filename,'right'),
                'file')
    footer_html = div('gordon.ball','footer')
    
    head_html = elem('head',elem('title',data.process().name_()))
    
    style_html = elem('style',
    """
    .title{font-weight:bold}
    .label{color:grey}
    .header{position:fixed;top:0px;left:0px;width:100%;background:#33cc00;font-weight:bold;font-size:120%}
    .footer{position:fixed;bottom:0px;left:0px;width:100%;background:#33cc00;text-align:right}
    .canvas{padding:40px 10px 40px 10px}
    .file{position:relative;background:#bbb;width:100%}
    .right{position:absolute;right:5px}
    .sequence{padding:1px 1px 1px 5px;border:1px solid #aaa}
    .sequence:hover{border 1px solid #00ffff}
    .sequence_label{background:lightskyblue}
    .sequence_label:hover{background:#fff}
    .sequence_area{display:block;padding:5px 0px 5px 5px}
    .edproducer{border:1px solid red;background-image:url('edproducer.png');background-position:center left;background-repeat:no-repeat;padding:0px 0px 0px 40px}
    .edfilter{border:1px solid green;background-image:url('edfilter.png');background-position:center left;background-repeat:no-repeat;padding:0px 0px 0px 40px}
    .edanalyzer{border:1px solid blue;background-image:url('edanalyzer.png');background-position:center left;background-repeat:no-repeat;padding:0px 0px 0px 40px}
    .outputmodule{border:1px solid green;background-image:url('outputmodule.png');background-position:center left;background-repeat:no-repeat;padding:0px 0px 0px 40px}
    .module{}
    .module_label:hover{background:#ccc}
    .module_area{display:none;padding:5px 0px 15px 15px;background:beige}
    .pset{border-spacing:10px 1px;border:1px solid black}
    .pset-item{}
    .pset-list{list-style-type:none;margin:0px;padding:2px 2px 2px 2px;border:1px solid grey}
    .pset-list-item{border-top:1px solid lightgrey;border-bottom:1px solid lightgrey}
    .param-name{font-weight:bold}
    .param-class{color:grey}
    .param-value-int{font-family:courier;color:blue}
    .param-value-double{font-family:courier;color:purple}
    .param-value-string{font-family:courier;color:brown}
    .param-value-bool{font-family:courier;color:#f0f}
    .param-value-inputtag{font-family:courier;color:red}
    .param-value-other{font-family:courier}
    .path{}
    """,
    type='text/css')
    
    script_html = elem('script',
    """
    function toggleVisible(id) {
      var elem = document.getElementById(id);
      if (elem.style.display=='block') {
        elem.style.display='none';
      } else {
        elem.style.display='block';      
      }
    }
    """,
    type='text/javascript')
    
    body_html = elem('body',script_html+header_html+footer_html+div(file_html+path_html,'canvas'))
    
    return elem('html',head_html+style_html+body_html)
    
  def export(self,data,filename,filetype):
    if not data.process():
      raise "HTMLExport requires a cms.Process object"
    
    html = self.produce(data)
    
    if filetype=='html':
      htmlfile = open(filename,'w')
      htmlfile.write(html)
      htmlfile.close()