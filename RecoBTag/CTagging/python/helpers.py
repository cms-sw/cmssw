import os
from RecoBTag.CTagging.trainingvars import get_var_pset
import xml.etree.ElementTree as ET
from pdb import set_trace

def get_path(file_in_path):
   'mimics edm.FileInPath behavior'
   search_env = os.environ.get('CMSSW_SEARCH_PATH', '')
   if not search_env:
      raise RuntimeError('The environmental variable CMSSW_SEARCH_PATH must be set')
   search_paths = search_env.split(':')
   for spath in search_paths:
      full_path = os.path.join(spath, file_in_path)
      if os.path.isfile(full_path):
         return full_path
   raise RuntimeError('No suitable path found for %s' % file_in_path)

def get_vars(xml_path, useFileInPath=True):
   full_path = get_path(xml_path) if useFileInPath else xml_path
   xml_tree = ET.parse(full_path)
   root = xml_tree.getroot()
   variables = None
   for i in root:
      if i.tag == 'Variables':
         variables = i

   if i is None:
      raise RuntimeError('Could not find Variables inside the xml weights')
   
   var_names = [i.attrib['Title'] for i in variables]
   return [get_var_pset(i) for i in var_names]
   
