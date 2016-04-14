import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import * 
GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_CONDITIONS"
GlobalTag.pfnPrefix = cms.untracked.string("frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/")
GlobalTag.globaltag = "80X_dataRun2_Express_v0"
es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

# ===== auto -> Automatically get the GT string from current Tier0 configuration via a Tier0Das call.
#       This needs a valid proxy to access the cern.ch network from the .cms one.
# 
auto=True
tier0DasUrl = 'https://cmsweb.cern.ch/t0wmadatasvc/prod/'

import os
import json
import sys, urllib2
import time
import ast

class Tier0DasInterface:
    """
    Class handling common Tier0-DAS queries and connected utilities
    """
    def __init__(self, url = 'https://cmsweb.cern.ch/t0wmadatasvc/prod/', proxy = None ):
        """
        Need base url for Tier0-DAS as input
        """
        self._t0DasBaseUrl = url
        self._debug = False
        self._retry = 0
        self._maxretry = 5
        self._proxy = proxy


    def getData(self, src, tout=5):
        """
        Get the JSON file for a give query specified via the Tier0-DAS url.
        Timeout can be set via paramter.
        """
        # actually get the json file from the given url of the T0-Das service
        # and returns the data

        try:
            if self._proxy:
                print "setting proxy"
                opener = urllib2.build_opener(urllib2.HTTPHandler(),
                                              urllib2.HTTPSHandler(),
                                              urllib2.ProxyHandler({'http':self._proxy, 'https':self._proxy}))
                urllib2.install_opener(opener)
            req = urllib2.Request(src)
            req.add_header("User-Agent",
                           "DQMIntegration/1.0 python/%d.%d.%d" % sys.version_info[:3])
            req.add_header("Accept","application/json")
            jsonCall = urllib2.urlopen(req, timeout = tout)
            url = jsonCall.geturl()
        except urllib2.HTTPError as  error:
            #print error.url
            errStr = "Cannot retrieve Tier-0 DAS data from URL \"" + error.url + "\""
            if self._proxy:
                errStr += " using proxy \"" + self._proxy + "\""
            print errStr
            print error
            raise urllib2.HTTPError("FIXME: handle exceptions")
        except urllib2.URLError as  error:
            if self._retry < self._maxretry:
                print 'Try # ' + str(self._retry) + " connection to Tier-0 DAS timed-out"
                self._retry += 1
                newtout = tout*self._retry
                time.sleep(3*self._retry)
                return self.getData(src,newtout)
            else:
                errStr = "Cannot retrieve Tier-0 DAS data from URL \"" + src + "\""
                if self._proxy:
                    errStr += " using proxy \"" + self._proxy + "\""
                self._retry = 0
                print errStr
                print error
                raise urllib2.URLError('TimeOut reading ' + src)

        except:
            raise
        else:
            if self._debug:
                print url
            jsonInfo = jsonCall.info()
            if self._debug:
                print jsonInfo
            jsonText = jsonCall.read()
            data = json.loads(jsonText)
            if self._debug:
                print "data:", data
            return data

    def getResultList(self, json):
        """
        Extractt the result list out of the JSON file
        """
        resultList = []
        #FIXME try
        resultList = json['result']

        if 'null' in resultList[0]:
            resultList[0] = resultList[0].replace('null','None')

        #print self.getValues(json, 'result')
        return resultList

    def getValues(self, json, key, selection=''):
        """
        Extract the value corrisponding to a given key from a JSON file. It is also possible to apply further selections.
        """
        # lookup for a key in a json file applying possible selections
        data = []
        check = 0
        if selection != '':
            check = 1
            (k, v) = selection

        for o in json:
            #print o
            try:
                if check == 1:
                    if (o[k] == v):
                        data.append(o[key])
                else:
                    data.append(o[key])
            except KeyError as error:
                print "[Tier0DasInterface::getValues] key: " + key + " not found in json file"
                print error
                raise
            except:
                print "[Tier0DasInterface::getValues] unknown error"
                raise
                #pass
        #print data
        return data

    def lastPromptRun(self):
        """
        Query to get the last run released for prompt
        """
        url = self._t0DasBaseUrl + "reco_config"
        try:
            json = self.getData(url)
            results = self.getResultList(json)
            workflowlist = ast.literal_eval(results[0])
            maxRun = -1
            for workflow in workflowlist:
                run = workflow['run']
                if int(run) > maxRun:
                    maxRun = run
            return maxRun
        except:
            print "[Tier0DasInterface::lastPromptRun] error"
            raise
            return 0

    def firstConditionSafeRun(self):
        """
        Query to ge the run for which the Tier0 system considers safe the update to the conditions
        """
        url = self._t0DasBaseUrl + "firstconditionsaferun"
        try:
            json = self.getData(url)
            results = self.getResultList(json)
            return results[0]
        except Exception as details:
            print "[Tier0DasInterface::firstConditionSafeRun] error", details
            raise
        return 0

    def promptGlobalTag(self, dataset):
        """
        Query the GT currently used by prompt = GT used by the last run released for prompt.
        """
        url = self._t0DasBaseUrl + "reco_config"
        #print "url =", url
        try:
            json = self.getData(url)
            results = self.getResultList(json)
            workflowlist = ast.literal_eval(results[0])
            gt = "UNKNOWN"
            for workflow in workflowlist:
                if workflow['primary_dataset'] == dataset:
                    gt = workflow['global_tag']
            # FIXME: do we realluy need to raise?
            if gt == "UNKNOWN":
                raise KeyError
            return gt
        except:
            print "[Tier0DasInterface::promptGlobalTag] error"
            raise
            return None

    def expressGlobalTag(self):
        """
        Query the GT currently used by express = GT used by the last run released for express.
        """
        url = self._t0DasBaseUrl + "express_config"
        #print "url =", url
        try:
            gt = "UNKNOWN"
            json = self.getData(url)
            results = self.getResultList(json)
            config = results[0]
            gt = str(config['global_tag'])
            # FIXME: do we realluy need to raise?
            if gt == "UNKNOWN":
                raise KeyError
            return gt
        except:
            print "[Tier0DasInterface::expressGlobalTag] error"
            raise
            return None


if auto:
    expressGT = "UNKNOWN"

    proxyurl = None
    if 'http_proxy' in os.environ:
        proxyurl = os.environ['http_proxy']
    test = Tier0DasInterface(url=tier0DasUrl,proxy = proxyurl)

    
    try:
        expressGT = test.expressGlobalTag()
        print "Tier0 DAS express GT:       ", expressGT
    except Exception as error:
        print 'Error'
        print error

    GlobalTag.globaltag = expressGT

