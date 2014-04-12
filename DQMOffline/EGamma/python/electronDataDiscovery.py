
#===================================================================
# So to get the list of input files. One must call :
#   search(), to get the list of primary files
#   search2(), to get the list of eventual secondary files
# 
# The selection of files is configured thanks to shell
# environment variables: 
# 
#   DD_RELEASE, for example CMSSW_2_2_0_pre1
#   DD_SAMPLE, for example RelValSingleElectronPt35
#   DD_RUN, for example ''
#   DD_COND , for example MC_31X_V2-v1
#   DD_TIER , for example RECO
#   DD_TIER_SECONDARY, for eventual secondary files
#   
#   DD_SOURCE:
#     das: use das
#     dbs: use dbs search
#     lsf: use dbs lsf
#     /castor/cern.ch/cms/...: assumed to be the path of a castor directory containing the input data files
#       for relvals: '/castor/cern.ch/cms/store/relval/${DD_RELEASE}/${DD_SAMPLE}/${DD_TIER}/${DD_COND}/'
#       for harvested dqm: '/castor/cern.ch/cms/store/unmerged/dqm/${DD_SAMPLE}-${DD_RELEASE}-${DD_COND}-DQM-DQMHarvest-OfflineDQM'
#     /eos/cms/...: assumed to be the path of a castor directory containing the input data files
#       for relvals: '/eos/cms/store/relval/${DD_RELEASE}/${DD_SAMPLE}/${DD_TIER}/${DD_COND}/'
#       for harvested dqm: '/eos/cms/store/unmerged/dqm/${DD_SAMPLE}-${DD_RELEASE}-${DD_COND}-DQM-DQMHarvest-OfflineDQM'
#     /...: assumed to be the path of a text file containing the list of input data files
#
# All except DD_SOURCE can use wildcard *.
#===================================================================

#import httplib, urllib, urllib2, types, string, os, sys
import os, sys, re, das_client

if not os.environ.has_key('DD_SOURCE'):
  os.environ['DD_SOURCE'] = 'das'
if not os.environ.has_key('DD_RELEASE'):
  os.environ['DD_RELEASE'] = ''
if not os.environ.has_key('DD_SAMPLE'):
  os.environ['DD_SAMPLE'] = ''
if not os.environ.has_key('DD_COND'):
  os.environ['DD_COND'] = ''
if not os.environ.has_key('DD_TIER'):
  os.environ['DD_TIER'] = ''
if not os.environ.has_key('DD_TIER_SECONDARY'):
  os.environ['DD_TIER_SECONDARY'] = ''
if not os.environ.has_key('DD_RUN'):
  os.environ['DD_RUN'] = ''
  
dd_release_re = re.compile(os.environ['DD_RELEASE'].replace('*','.*')) ;
dd_sample_re = re.compile(os.environ['DD_SAMPLE'].replace('*','.*')) ;
dd_cond_re = re.compile(os.environ['DD_COND'].replace('*','.*')) ;
dd_run_re = re.compile(os.environ['DD_RUN'].replace('*','.*')) ;

def common_search(dd_tier):

  dd_tier_re = re.compile(dd_tier.replace('*','.*')) ;

  if os.environ['DD_SOURCE'] == "das":
  
    query = "dataset instance=cms_dbs_prod_global"
    if os.environ['DD_RELEASE'] != "" :
      query = query + " release=" + os.environ['DD_RELEASE']
    if os.environ['DD_SAMPLE'] != "":
      query = query + " primary_dataset=" + os.environ['DD_SAMPLE']
    if dd_tier != "":
      query = query + " tier=" + dd_tier
    if os.environ['DD_COND'] != "":
      query = query + " dataset=*" + os.environ['DD_COND'] + "*"
    if os.environ['DD_RUN'] != "":
      query = query + " run=" + os.environ['DD_RUN']
    #query = query + " | unique" # too long ??
    
    #data = os.popen('das_client.py --limit=0 --query "'+query+'"')
    #datalines = data.readlines()
    #data.close()
    #datasets = []
    #for line in datalines:
    #  line = line.rstrip()
    #  if line != "" and line[0] =="/":
    #    datasets.append(line)
    #dataset = datasets[0]
    
    data = das_client.json.loads(das_client.get_data('https://cmsweb.cern.ch',query,0,0,0))
            
    if data['nresults']==0:
      print '[electronDataDiscovery.py] No DAS dataset for query:', query
      return []
    while data['nresults']>1:
      if data['data'][0]['dataset'][0]['name']==data['data'][1]['dataset'][0]['name']:
        data['data'].pop(0)
        data['nresults'] -= 1
      else:
        print '[electronDataDiscovery.py] Several DAS datasets for query:', query
        for i in range(data['nresults']):
          print '[electronDataDiscovery.py] dataset['+str(i)+']: '+data['data'][i]['dataset'][0]['name']
        return []

    dataset = data['data'][0]['dataset'][0]['name']
    
    query = "file instance=cms_dbs_prod_global dataset="+dataset
    
    #data = os.popen('das_client.py --limit=0 --query "'+query+'"')
    #datalines = data.readlines()
    #data.close()
    #result = []
    #for line in datalines:
    #  line = line.rstrip()
    #  if line != "" and line[0] =="/":
    #    result.append(line)
    
    data = das_client.json.loads(das_client.get_data('https://cmsweb.cern.ch',query,0,0,0))
    
    if data['nresults']==0:
      print '[electronDataDiscovery.py] No DAS file in dataset:', dataset
      return []
      
    result = []
    for i in range(0,data['nresults']):
      result.append(str(data['data'][i]['file'][0]['name']))
    
  elif os.environ['DD_SOURCE'] == "dbs":
  
    input = "find file"
    separator = " where "
    if os.environ['DD_RELEASE'] != "":
      input = input + separator + "release = " + os.environ['DD_RELEASE']
      separator = " and "
    if os.environ['DD_SAMPLE'] != "":
      input = input + separator + "primds = " + os.environ['DD_SAMPLE']
      separator = " and "
    if os.environ['DD_RUN'] != "":
      input = input + separator + "run = " + os.environ['DD_RUN']
      separator = " and "
    input = input + separator + "dataset like *" + os.environ['DD_COND'] + "*" + dd_tier + "*"
    
    data = os.popen('dbs search --url="http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet" --query "'+input+'"')
    datalines = data.readlines()
    data.close()
    result = []
    for line in datalines:
      line = line.rstrip()
      if line != "" and line[0] =="/":
        result.append(line)
    
  elif os.environ['DD_SOURCE'] == "http":
  
    input = "find file"
    separator = " where "
    if os.environ['DD_RELEASE'] != "":
      input = input + separator + "release = " + os.environ['DD_RELEASE']
      separator = " and "
    if os.environ['DD_SAMPLE'] != "":
      input = input + separator + "primds = " + os.environ['DD_SAMPLE']
      separator = " and "
    if os.environ['DD_RUN'] != "":
      input = input + separator + "run = " + os.environ['DD_RUN']
      separator = " and "
    input = input + separator + "dataset like *" + os.environ['DD_COND'] + "*" + dd_tier + "*"
    
    url = "https://cmsweb.cern.ch:443/dbs_discovery/aSearch"
    final_input = urllib.quote(input) ;
    
    agent   = "Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)"
    ctypes  = "text/plain"
    headers = { 'User-Agent':agent, 'Accept':ctypes}
    params  = {'dbsInst':'cms_dbs_prod_global',
               'html':0,'caseSensitive':'on','_idx':0,'pagerStep':-1,
               'userInput':final_input,
               'xml':0,'details':0,'cff':0,'method':'dbsapi'}
    data    = urllib.urlencode(params,doseq=True)
    req     = urllib2.Request(url, data, headers)
    data    = ""
    
    try:
      response = urllib2.urlopen(req)
      data = response.read()
    except urllib2.HTTPError, e:
      if e.code==201:
        print e.headers       
        print e.msg
        pass
      else:
        raise e

    datalines = data.readlines()
    data.close()
    result = []
    for line in datalines:
      line = line.rstrip()
      if line != "" and line[0] =="/":
        result.append(line)
    
  elif os.environ['DD_SOURCE'] == "lsf":
  
    dbs_path = '/'+os.environ['DD_SAMPLE']+'/'+os.environ['DD_RELEASE']+'-'+os.environ['DD_COND']+'/'+os.environ['DD_TIER']+'"'
    if __name__ == "__main__":
      print 'dbs path:',dbs_path
    data = os.popen('dbs lsf --path="'+dbs_path+'"')
    datalines = data.readlines()
    data.close()
    result = []
    for line in datalines:
      line = line.rstrip()
      if line != "" and line[0] =="/":
        result.append(line)
      
  elif os.environ['DD_SOURCE'].startswith('/castor/cern.ch/cms/'): # assumed to be a castor dir
  
    castor_dir = os.environ['DD_SOURCE'].replace('/castor/cern.ch/cms/','/',1)
    result = []
    data = os.popen('rfdir /castor/cern.ch/cms'+castor_dir)
    subdirs = data.readlines()
    data.close()
    datalines = []
    for line in subdirs:
      line = line.rstrip()
      subdir = line.split()[8]
      data = os.popen('rfdir /castor/cern.ch/cms'+castor_dir+'/'+subdir)
      datalines = data.readlines()
      for line in datalines:
        line = line.rstrip()
        file = line.split()[8]
        if file != "":
          result.append(castor_dir+'/'+subdir+'/'+file)
      data.close()
      
  elif os.environ['DD_SOURCE'].startswith('/eos/cms/'): # assumed to be an eos dir
  
    data = os.popen('/afs/cern.ch/project/eos/installation/pro/bin/eos.select find -f '+os.environ['DD_SOURCE'])
    lines = data.readlines()
    data.close()
    result = []
    for line in lines:
      line = line.strip().replace('/eos/cms/','/',1)
      if line == "": continue
      if dd_sample_re.search(line) == None: continue
      if dd_cond_re.search(line) == None: continue
      if dd_tier_re.search(line) == None: continue
      if dd_run_re.search(line) == None: continue
      result.append(line)
      
  else: # os.environ['DD_SOURCE'] is assumed to be a file name
  
    result = []
    for line in open(os.environ['DD_SOURCE']).readlines():
      line = os.path.expandvars(line.strip())
      if line == "": continue
      if dd_sample_re.search(line) == None: continue
      if dd_cond_re.search(line) == None: continue
      if dd_tier_re.search(line) == None: continue
      if dd_run_re.search(line) == None: continue
      result.append(line)
      
    if len(result)==0:
      diag = '[electronDataDiscovery.py] No more files after filtering with :'
      if os.environ['DD_SAMPLE']!='': diag += ' ' + os.environ['DD_SAMPLE']
      if os.environ['DD_COND']!='': diag += ' ' + os.environ['DD_COND']
      if dd_tier!='': diag += ' ' + dd_tier
      if os.environ['DD_RUN']!='': diag += ' ' + os.environ['DD_RUN']
      print diag
      
  return result

def search():
  return common_search(os.environ['DD_TIER'])

def search2():
  return common_search(os.environ['DD_TIER_SECONDARY'])

	
	

