#!/usr/bin/env python

import httplib, urllib, urllib2, types, string, os, sys

# return the list of files obtained from the data discovery and based upon environnement variables:
# DBS_RELEASE, for example CMSSW_2_2_0_pre1
# DBS_SAMPLE, for example RelValSingleElectronPt35
# DBS_LIKE , for example *RECO

def search():

  if os.environ['DBS_RELEASE'] == "LOCAL":
    if os.environ['DBS_SAMPLE'] == "RelValSingleElectronPt10":
      result = ['file:../../Configuration/test/SingleElectronPt10Reco.root']
    elif os.environ['DBS_SAMPLE'] == "RelValSingleElectronPt35":
      result = ['file:../../Configuration/test/SingleElectronPt35Reco.root']
    elif os.environ['DBS_SAMPLE'] == "RelValQCD_Pt_80_120":
      result = ['file:../../Configuration/test/QcdPt80-120Reco.root']
    else:
      result = ['NONE']
  else:
    url = "https://cmsweb.cern.ch:443/dbs_discovery/aSearch"
    if os.environ['DBS_RELEASE'] != "Any":
      input = "find file where release = " + os.environ['DBS_RELEASE']
    if os.environ['DBS_SAMPLE'] != "Any":
      input = input + " and primds = " + os.environ['DBS_SAMPLE']
    input = input + " and dataset like " + os.environ['DBS_LIKE']
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

    result = []
    for line in data.split("\n"):
      if line != "" and line[0] =="/":
        result.append(line)

  return result

if __name__ == "__main__":
  for file in search():
    print file

	
	

