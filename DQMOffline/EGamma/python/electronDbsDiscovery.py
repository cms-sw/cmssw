
#===================================================================
# This python module is querying https://cmsweb.cern.ch/dbs_discovery/
# so to get the list of input files. One must call :
#   search(), to get the list of primary files
#   search2(), to get the list of eventual secondary files
# 
# The selection of files is configured thanks to shell
# environment variables: 
# 
#   DBS_RELEASE, for example CMSSW_2_2_0_pre1
#   DBS_SAMPLE, for example RelValSingleElectronPt35
#   DBS_RUN, for example Any
#   DBS_COND , for example MC_31X_V2-v1
#   DBS_TIER , for example RECO
#   DBS_TIER_SECONDARY, for eventual secondary files
#   
#   DBS_STRATEGY:
#     search: use dbs search
#     castor: use rfdir
#     lsf: use dbs lsf
#     local: look within electronDbsDiscovery.txt
#   DBS_CASTOR_DIR: top dir to scrutiny when the strategy is "castor"
#     for relvals: '/store/relval/${DBS_RELEASE}/${DBS_SAMPLE}/${DBS_TIER}/${DBS_COND}/'
#     for harvested dqm: '/store/unmerged/dqm/${DBS_SAMPLE}-${DBS_RELEASE}-${DBS_COND}-DQM-DQMHarvest-OfflineDQM'
#
# In DBS_COND, DBS_TIER and DBS_TIER_SECONDARY,
# one can use wildcard *.
#===================================================================

#import httplib, urllib, urllib2, types, string, os, sys
import os, sys

if not os.environ.has_key('DBS_STRATEGY'):
  os.environ['DBS_STRATEGY'] = "search"
if not os.environ.has_key('DBS_RELEASE'):
  os.environ['DBS_RELEASE'] = "Any"
if not os.environ.has_key('DBS_SAMPLE'):
  os.environ['DBS_SAMPLE'] = "Any"
if not os.environ.has_key('DBS_RUN'):
  os.environ['DBS_RUN'] = "Any"
if not os.environ.has_key('DBS_TIER_SECONDARY'):
  os.environ['DBS_TIER_SECONDARY'] = ""

def common_search(dbs_tier):

  if os.environ['DBS_STRATEGY'] == "local":
  
    result = []
    for line in  open('electronDbsDiscovery.txt').readlines():
      line = os.path.expandvars(line.strip())
      if line == "": continue
      if os.environ['DBS_SAMPLE'] != "Any" and line.find(os.environ['DBS_SAMPLE'])== -1: continue
      if line.find(os.environ['DBS_COND'])== -1: continue
      if line.find(dbs_tier)== -1: continue
      result.append('file:'+line)
      
  elif os.environ['DBS_STRATEGY'] == "castor":
  
    castor_dir = os.environ['DBS_CASTOR_DIR']
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
      
  elif os.environ['DBS_STRATEGY'] == "lsf":
  
    dbs_path = '/'+os.environ['DBS_SAMPLE']+'/'+os.environ['DBS_RELEASE']+'-'+os.environ['DBS_COND']+'/'+os.environ['DBS_TIER']+'"'
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
      
  else:
  
    input = "find file"
    separator = " where "
    if os.environ['DBS_RELEASE'] != "Any":
      input = input + separator + "release = " + os.environ['DBS_RELEASE']
      separator = " and "
    if os.environ['DBS_SAMPLE'] != "Any":
      input = input + separator + "primds = " + os.environ['DBS_SAMPLE']
      separator = " and "
    if os.environ['DBS_RUN'] != "Any":
      input = input + separator + "run = " + os.environ['DBS_RUN']
      separator = " and "
    input = input + separator + "dataset like *" + os.environ['DBS_COND'] + "*" + dbs_tier + "*"
    
    #url = "https://cmsweb.cern.ch:443/dbs_discovery/aSearch"
    #final_input = urllib.quote(input) ;
    #
    #agent   = "Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)"
    #ctypes  = "text/plain"
    #headers = { 'User-Agent':agent, 'Accept':ctypes}
    #params  = {'dbsInst':'cms_dbs_prod_global',
    #           'html':0,'caseSensitive':'on','_idx':0,'pagerStep':-1,
    #           'userInput':final_input,
    #           'xml':0,'details':0,'cff':0,'method':'dbsapi'}
    #data    = urllib.urlencode(params,doseq=True)
    #req     = urllib2.Request(url, data, headers)
    #data    = ""
    #
    #try:
    #  response = urllib2.urlopen(req)
    #  data = response.read()
    #except urllib2.HTTPError, e:
    #  if e.code==201:
    #    print e.headers       
    #    print e.msg
    #    pass
    #  else:
    #    raise e

    data = os.popen('dbs search --url="http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet" --query "'+input+'"')
    datalines = data.readlines()
    data.close()
    result = []
    for line in datalines:
      line = line.rstrip()
      if line != "" and line[0] =="/":
        result.append(line)
    
  return result

def search():
  return common_search(os.environ['DBS_TIER'])

def search2():
  return common_search(os.environ['DBS_TIER_SECONDARY'])

	
	

