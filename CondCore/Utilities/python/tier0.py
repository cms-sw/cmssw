
#-toDo: move this to common?

import logging
import json
import os
import sys
import time
import subprocess

import pycurl

tier0Url = os.getenv('TIER0_API_URL', 'https://cmsweb.cern.ch/t0wmadatasvc/prod/')

class Tier0Error(Exception):
    '''Tier0 exception.
    '''

    def __init__(self, message):
        self.args = (message, )


def unique(seq, keepstr=True):
    t = type(seq)
    if t is str:
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

#note: this exception seems unused
class ResponseError( Tier0Error ):

    def __init__( self, curl, response, proxy, timeout, maxTime ):
        super( ResponseError, self ).__init__( response )
        self.args += ( curl, proxy )
        self.timeout = timeout
        self.maxTime = maxTime

    def __str__(self):
        errStr = f'Wrong response for curl connection to Tier0DataSvc'\
                 f' from URL "{self.args[1].getinfo(self.args[1].EFFECTIVE_URL)}"'
        if self.args[-1]:
            errStr += f' using proxy "{str(self.args[-1])}"'
        errStr += f' with connection-timeout "{self.timeout}", max-time "{self.maxtime}"'\
                  f' with error code "{self.args[1].getinfo(self.args[1].RESPONSE_CODE)}".'
        if '<p>' in self.args[0]:
            full_response = self.args[0].partition('<p>')[-1].rpartition('</p>')[0]
            errStr += f'\nFull response: "{full_response}".'
        else:
            errStr += f'\nFull response: "{self.args[0]}".'
        
        return errStr

#TODO: Add exceptions for each category of HTTP error codes
#TODO: check response code and raise corresponding exceptions
#note: this function seems to be unused
def _raise_http_error( curl, response, proxy, timeout, maxTime ):
    raise ResponseError( curl, response, proxy, timeout, maxTime )

class Tier0Handler( object ):

    def __init__( self, uri, timeOut, maxTime, retries, retryPeriod, proxy, debug ):
        """
        Parameters:
        uri: Tier0DataSvc URI;
        timeOut: time out for connection of Tier0DataSvc HTTPS calls [seconds];
        maxTime: maximum time for Tier0DataSvc HTTPS calls (including data transfer) [seconds];
        retries: maximum retries for Tier0DataSvc HTTPS calls;
        retryPeriod: sleep time between two Tier0DataSvc HTTPS calls [seconds];
        proxy: HTTP proxy for accessing Tier0DataSvc HTTPS calls;
        debug: if set to True, enables debug information.
        """
        self._uri = uri
        self._timeOut = timeOut
        self._maxTime = maxTime
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

    def _getCerts( self ) -> str:
        cert_path = os.getenv('X509_USER_CERT', '')
        key_path = os.getenv('X509_USER_KEY', '')
        
        certs = ""
        if cert_path:
            certs += f' --cert {cert_path}'
        else:
            logging.warning("No certificate provided for Tier0 access, use X509_USER_CERT and"
                            " optionally X509_USER_KEY env variables to specify the path to the cert"
                            " (and the key unless included in the cert file)")
        if key_path:
            certs += f' --key {key_path}'
        return certs

    def _curlQueryTier0( self, url:str, force_debug:bool = False, force_cert:bool = False):
        userAgent = "User-Agent: ConditionWebServices/1.0 python/%d.%d.%d PycURL/%s" \
            % ( sys.version_info[ :3 ] + ( pycurl.version_info()[ 1 ], ) )
        debug = "-v" if self._debug or force_debug else "-s -S"

        proxy = f"--proxy {self._proxy}" if self._proxy else ""
        certs = self._getCerts() if not self._proxy or force_cert else ""
        
        cmd = f'/usr/bin/curl -k -L --user-agent "{userAgent}" {proxy}'\
              f' --connect-timeout {self._timeOut} --max-time {self._maxTime} --retry {self._retries}'\
              f' {debug} {url} {certs}'

        # time the curl to understand if re-tries have been carried out
        start = time.time()
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdoutdata, stderrdata) =  process.communicate()
        end = time.time()
        return process.returncode, stdoutdata, stderrdata, end-start

    def _queryTier0DataSvc( self, url ):
        """
        Queries Tier0DataSvc.
        url: Tier0DataSvc URL.
        @returns: dictionary, from whence the required information must be retrieved according to the API call.
        Raises if connection error, bad response, or timeout after retries occur.
        """

        retcode, stdoutdata, stderrdata, query_time = self._curlQueryTier0(url)

        if retcode != 0 or stderrdata:

            # if the first curl has failed, logg its stderror and prepare and independent retry
            msg = "looks like curl returned an error: retcode=%s and took %s seconds" % (retcode, query_time,)
            msg += ' msg = "'+str(stderrdata)+'"'
            logging.error(msg)
            if self._proxy:
                logging.info("before assumed proxy provides authentication, now trying with both proxy and certificate")
                
            time.sleep(self._retryPeriod)
            retcode, stdoutdata, stderrdata, query_time = self._curlQueryTier0(url, force_debug=True, force_cert=True)
            if retcode != 0:
                msg = "looks like curl returned an error for the second time: retcode=%s" % (retcode,)
                msg += ' msg = "'+str(stderrdata)+'"'
                logging.error(msg)
                raise Tier0Error(msg)
            else:
                msg = "curl returned ok upon the second try"
                logging.info(msg)
        resp = json.loads( ''.join(stdoutdata.decode()).replace( "'", '"').replace(' None', ' "None"') )
        return resp


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
            errStr = """First condition safe run is not available in Tier0DataSvc from URL \"%s\"""" \
                %( os.path.join( self._uri, firstConditionSafeRunAPI ), )
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
        gtnames = sorted(unique( [ str( di['global_tag'] ) for di in data['result'] if di['global_tag'] is not None ] ))
        try:
            recentGT = gtnames[-1]
            return recentGT
        except IndexError:
            errStr = """No Global Tags for \"%s\" are available in Tier0DataSvc from URL \"%s\"""" \
                %( config, os.path.join( self._uri, config ) )
            if self._proxy:
                errStr += """ using proxy \"%s\".""" %( str( self._proxy ), )
        raise Tier0Error( errStr )


def test( url ):
    t0 = Tier0Handler( url, 1, 5, 1, 10, None, debug=False)

    print('   fcsr = %s (%s)' % (t0.getFirstSafeRun(), type(t0.getFirstSafeRun()) ))
    print('   reco_config = %s' % t0.getGlobalTag('reco_config'))
    print('   express_config = %s' % t0.getGlobalTag('express_config'))
    print('\n')


if __name__ == '__main__':
    test( tier0Url )
