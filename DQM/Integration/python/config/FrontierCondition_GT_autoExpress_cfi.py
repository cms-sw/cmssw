from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *

# Default Express GT: it is the GT that will be used in case we are not able
# to retrieve the one used at Tier0.
# It should be kept in synch with Express processing at Tier0: what the url
# https://cmsweb.cern.ch/t0wmadatasvc/prod/express_config
# would tell you.
GlobalTag.globaltag = "102X_dataRun2_Express_v1"

# ===== auto -> Automatically get the GT string from current Tier0 configuration via a Tier0Das call.
#       This needs a valid proxy to access the cern.ch network from the .cms one.
#
auto=False

# The implementation of the class is reused from the condition upload service.
#TODO: make this class a common utility under Conditions or Config.DP
import json
import os
import pycurl
import subprocess
import sys
import time

tier0Url = 'https://cmsweb.cern.ch/t0wmadatasvc/prod/'

class Tier0Error(Exception):
    '''Tier0 exception.
    '''

    def __init__(self, message):
        self.args = (message, )

def unique(seq, keepstr=True):
    t = type(seq)
    if t in (unicode, str):
        t = (list, t('').join)[bool(keepstr)]
    try:
        remaining = set(seq)
        seen = set()
        return t(c for c in seq if (c in remaining and not remaining.remove(c)))
    except TypeError: # hashing didn't work, see if seq is sortable
        try:
            from itertools import groupby
            s = sorted(enumerate(seq),key=lambda i_v1:(i_v1[1],i_v1[0]))
            return t(next(g) for k,g in groupby(s, lambda i_v: i_v[1]))
        except:  # not sortable, use brute force
            seen = []
            return t(c for c in seq if not (c in seen or seen.append(c)))

class Tier0Handler( object ):

    def __init__( self, uri, timeOut, retries, retryPeriod, proxy, debug ):
        """
        Parameters:
        uri: Tier0DataSvc URI;
        timeOut: time out for Tier0DataSvc HTTPS calls;
        retries: maximum retries for Tier0DataSvc HTTPS calls;
        retryPeriod: sleep time between two Tier0DataSvc HTTPS calls;
        proxy: HTTP proxy for accessing Tier0DataSvc HTTPS calls;
        debug: if set to True, enables debug information.
        """
        self._uri = uri
        self._timeOut = timeOut
        self._retries = retries
        self._retryPeriod = retryPeriod
        self._proxy = proxy
        self._debug = debug

    def setDebug( self ):
        self._debug = True

    def unsetDebug( self ):
        self._debug = False

    def setProxy( self, proxy ):
        self._proxy = proxy

    def _queryTier0DataSvc( self, url ):
        """
        Queries Tier0DataSvc.
        url: Tier0DataSvc URL.
        @returns: dictionary, from whence the required information must be retrieved according to the API call.
        Raises if connection error, bad response, or timeout after retries occur.
        """

        userAgent = "User-Agent: DQMIntegration/2.0 python/%d.%d.%d PycURL/%s" % ( sys.version_info[ :3 ] + ( pycurl.version_info()[ 1 ], ) )

        proxy = ""
        if self._proxy: proxy = ' --proxy %s ' % self._proxy
        
        debug = " -s -S "
        if self._debug: debug = " -v "
        
        cmd = '/usr/bin/curl -k -L --user-agent "%s" %s --connect-timeout %i --retry %i %s %s ' % (userAgent, proxy, self._timeOut, self._retries, debug, url)

        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdoutdata, stderrdata) =  process.communicate()
        retcode = process.returncode

        if retcode != 0 or stderrdata:
           msg = "looks like curl returned an error: retcode=%s" % (retcode,)
           msg += ' msg = "'+str(stderrdata)+'"'
           raise Tier0Error(msg)

        return json.loads( ''.join(stdoutdata).replace( "'", '"').replace(' None', ' "None"') )

    def getFirstSafeRun( self ):
        """
        Queries Tier0DataSvc to get the first condition safe run.
        Parameters:
        @returns: integer, the run number.
        Raises if connection error, bad response, timeout after retries occur, or if the run number is not available.
        """
        firstConditionSafeRunAPI = "firstconditionsaferun"
        safeRunDict = self._queryTier0DataSvc( os.path.join( self._uri, firstConditionSafeRunAPI ) )
        if safeRunDict is None:
            errStr = """First condition safe run is not available in Tier0DataSvc from URL \"%s\" """ %( os.path.join( self._uri, firstConditionSafeRunAPI ), )
            if self._proxy:
                errStr += """ using proxy \"%s\".""" %( str( self._proxy ), )
            raise Tier0Error( errStr )
        return int(safeRunDict['result'][0])

    def getGlobalTag( self, config ):
        """
        Queries Tier0DataSvc to get the most recent Global Tag for a given workflow.
        Parameters:
        config: Tier0DataSvc API call for the workflow to be looked for;
        @returns: a string with the Global Tag name.
        Raises if connection error, bad response, timeout after retries occur, or if no Global Tags are available.
        """
        data = self._queryTier0DataSvc( os.path.join( self._uri, config ) )
        gtnames = sorted(unique( [ str( di[ 'global_tag' ] ) for di in data['result'] if di[ 'global_tag' ] is not None ] ))
        try:
            recentGT = gtnames[-1]
            return recentGT
        except IndexError:
            errStr = """No Global Tags for \"%s\" are available in Tier0DataSvc from URL \"%s\" """ %( config, os.path.join( self._uri, config ) )
            if self._proxy:
                errStr += """ using proxy \"%s\".""" %( str( self._proxy ), )
        raise Tier0Error( errStr )

if auto:
    proxyurl = None
    if 'http_proxy' in os.environ:
        proxyurl = os.environ[ 'http_proxy' ]
    t0 = Tier0Handler( tier0Url, 5, 5, 5, proxyurl, False )

    try:
        # Get the express GT from Tie0 DataService API
        GlobalTag.globaltag = cms.string( t0.getGlobalTag( 'express_config' ) )
        print("The query to the Tier0 DataService returns the express GT: \"%s\"" % ( GlobalTag.globaltag.value(), ))
    except Tier0Error as error:
        # the web query did not succeed, fall back to the default
        print("Error in querying the Tier0 DataService")
        print(error)
        print("Falling back to the default value of the express GT: \"%s\"" % ( GlobalTag.globaltag.value(), ))
else:
    print("Using hardcoded GT: \"%s\"" % GlobalTag.globaltag.value())
