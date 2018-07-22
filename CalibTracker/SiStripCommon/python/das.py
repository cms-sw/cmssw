from __future__ import print_function
from das_client import get_data
import subprocess
#from pdb import set_trace

def query(query_str, verbose=False):
   'simple query function to interface with DAS, better than using Popen as everything is handled by python'
   if verbose:
      print('querying DAS with: "%s"' % query_str)
   data = get_data(
      'https://cmsweb.cern.ch', 
      query_str,
      0, 0, False)

   to_get = query_str.split()[0].strip(',')
   if data['status'] != 'ok':
      raise RuntimeError('Das query crashed')

   #-1 works both when getting dataset from files and files from datasets, 
   #not checked on everything
   return [i[to_get][-1]['name'] for i in data['data']]
