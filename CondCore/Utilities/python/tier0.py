from __future__ import print_function

#-toDo: move this to common?

import logging
import json
import os
import sys
import time
import subprocess

import pycurl

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

class ResponseError( Tier0Error ):

    def __init__( self, curl, response, proxy, timeout ):
        super( ResponseError, self ).__init__( response )
        self.args += ( curl, proxy )
        self.timeout = timeout

    def __str__( self ):
        errStr = """Wrong response for curl connection to Tier0DataSvc from URL \"%s\"""" %( self.args[1].getinfo( self.args[1].EFFECTIVE_URL ), )
        if self.args[ -1 ]:
            errStr += """ using proxy \"%s\"""" %( str( self.args[ -1 ] ), )
        errStr += """ with timeout \"%d\" with error code \"%d\".""" %( self.timeout, self.args[1].getinfo( self.args[1].RESPONSE_CODE) )
        if self.args[0].find( '<p>' ) != -1:
            errStr += """\nFull response: \"%s\".""" %( self.args[0].partition('<p>')[-1].rpartition('</p>')[0], )
        else:
            errStr += """\nFull response: \"%s\".""" %( self.args[0], )
        return errStr

#TODO: Add exceptions for each category of HTTP error codes
#TODO: check response code and raise corresponding exceptions

def _raise_http_error( curl, response, proxy, timeout ):
    raise ResponseError( curl, response, proxy, timeout )

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
        
        userAgent = "User-Agent: ConditionWebServices/1.0 python/%d.%d.%d PycURL/%s" % ( sys.version_info[ :3 ] + ( pycurl.version_info()[ 1 ], ) )

        proxy = ""
        if self._proxy: proxy = ' --proxy=%s ' % self._proxy
        
        debug = " -s -S "
        if self._debug: debug = " -v "
        
        cmd = '/usr/bin/curl -k -L --user-agent "%s" %s --connect-timeout %i --retry %i %s %s ' % (userAgent, proxy, self._timeOut, self._retries, debug, url)

        # time the curl to understand if re-tries have been carried out
        start = time.time()
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdoutdata, stderrdata) =  process.communicate()
        retcode = process.returncode
        end = time.time()

        if retcode != 0 or stderrdata:
           # if the first curl has failed, logg its stderror and prepare and independent retry
           msg = "looks like curl returned an error: retcode=%s and took %s seconds" % (retcode,(end-start),)
           msg += ' msg = "'+str(stderrdata)+'"'
           logging.error(msg)

           time.sleep(10)
           cmd = '/usr/bin/curl -k -L --user-agent "%s" %s --connect-timeout %i --retry %i %s %s ' % (userAgent, proxy, self._timeOut, self._retries, "-v", url)
           process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
           (stdoutdata, stderrdata) =  process.communicate()
           retcode = process.returncode
           if retcode != 0:
              msg = "looks like curl returned an error for the second time: retcode=%s" % (retcode,)
              msg += ' msg = "'+str(stderrdata)+'"'
              logging.error(msg)
              raise Tier0Error(msg)
           else :
              msg = "curl returned ok upon the second try"
              logging.info(msg)

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
            errStr = """First condition safe run is not available in Tier0DataSvc from URL \"%s\"""" %( os.path.join( self._uri, firstConditionSafeRunAPI ), )
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
            errStr = """No Global Tags for \"%s\" are available in Tier0DataSvc from URL \"%s\"""" %( config, os.path.join( self._uri, config ) )
            if self._proxy:
                errStr += """ using proxy \"%s\".""" %( str( self._proxy ), )
        raise Tier0Error( errStr )


def test( url ):
    t0 = Tier0Handler( url, 1, 1, 1, None, debug=False)

    print('   fcsr = %s (%s)' % (t0.getFirstSafeRun(), type(t0.getFirstSafeRun()) ))
    print('   reco_config = %s' % t0.getGlobalTag('reco_config'))
    print('   express_config = %s' % t0.getGlobalTag('express_config'))
    print('\n')


if __name__ == '__main__':
    test( tier0Url )

