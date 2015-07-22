#!/usr/bin/env python
'''Script that uploads to the new dropBox.
'''

__author__ = 'Miguel Ojeda'
__copyright__ = 'Copyright 2012, CERN CMS'
__credits__ = ['Giacomo Govi', 'Salvatore Di Guida', 'Miguel Ojeda', 'Andreas Pfeiffer']
__license__ = 'Unknown'
__maintainer__ = 'Miguel Ojeda'
__email__ = 'mojedasa@cern.ch'
__version__ = 6


import os
import sys
import logging
import optparse
import hashlib
import tarfile
import netrc
import getpass
import errno
import sqlite3
import json
import tempfile


defaultBackend = 'online'
defaultHostname = 'cms-conddb-prod.cern.ch'
defaultUrlTemplate = 'https://%s/dropBox/'
defaultTemporaryFile = 'upload.tar.bz2'
defaultNetrcHost = 'DropBox'
defaultWorkflow = 'offline'


# common/http.py start (plus the "# Try to extract..." section bit)
import re
import time
import logging
import cStringIO
import HTMLParser
import urllib

import pycurl
import copy


class CERNSSOError(Exception):
    '''A CERN SSO exception.
    '''


def _getCERNSSOCookies(url, secureTarget = True, secure = True):
    '''Returns the required CERN SSO cookies for a URL using Kerberos.

    They can be used with any HTTP client (libcurl, wget, urllib...).

    If you wish to make secure SSL connections to the CERN SSO
    (i.e. verify peers/hosts), you may need to install the CERN-CA-certs package.
    Use secure == False to skip this (i.e. this is the same as curl
    -k/--insecure). Not recommended: tell users to install them or use lxplus6.

    The same way, if you have a self-signed certificate in your target URL
    you can use secureTarget == False as well. Note that this option
    is provided in order to be able to use a secure SSL connection to CERN SSO,
    even if the connection to your target URL is not secure. Note that
    you will probably need the CERN-CA-certs package after you get a certificate
    signed by the CERN CA (https://cern.ch/ca), even if you did not need it
    for the CERN SSO.

    Note that this method *does* a query to the given URL if successful.

    This was implemented outside the HTTP class for two main reasons:

        * The only thing needed to use CERN SSO is the cookies, therefore
          this function is useful alone as well (e.g. as a simple replacement
          of the cern-get-sso-cookie script or as a Python port of
          the WWW::CERNSSO::Auth Perl package -- this one does not write
          any file and can be used in-memory, by the way).

        * We do not need to use the curl handler of the HTTP class.
          This way we do not overwrite any options in that one and we use
          only a temporary one here for getting the cookie.

    TODO: Support also Certificate/Key authentication.
    TODO: Support also Username/Password authentication.
    TODO: Review the error paths.
    TODO: Why PERLSESSID was used in the original code?
    TODO: Retry if timeouts are really common (?)
    '''

    def perform():
        response = cStringIO.StringIO()
        curl.setopt(curl.WRITEFUNCTION, response.write)
        curl.perform()
        code = curl.getinfo(curl.RESPONSE_CODE)
        response = response.getvalue()
        effectiveUrl = curl.getinfo(curl.EFFECTIVE_URL)
        return (code, response, effectiveUrl)

    # These constants and the original code came from the official CERN
    # cern-get-sso-cookie script and WWW::CERNSSO::Auth Perl package.
    VERSION = '0.4.2'
    CERN_SSO_CURL_USER_AGENT_KRB = 'curl-sso-kerberos/%s' % VERSION
    CERN_SSO_CURL_AUTHERR = 'HTTP Error 401.2 - Unauthorized'
    CERN_SSO_CURL_ADFS_EP = '/adfs/ls/auth'
    CERN_SSO_CURL_ADFS_SIGNIN = 'wa=wsignin1.0'
    CERN_SSO_CURL_CAPATH = '/etc/pki/tls/certs'

    logging.debug('secureTarget = %s', secureTarget)
    logging.debug('secure = %s', secure)

    curl = pycurl.Curl()

    # Store the cookies in memory, which we will retreive later on
    curl.setopt(curl.COOKIEFILE, '')

    # This should not be needed, but sometimes requests hang 'forever'
    curl.setopt(curl.TIMEOUT, 10)
    curl.setopt(curl.CONNECTTIMEOUT, 10)

    # Ask curl to use Kerberos5 authentication
    curl.setopt(curl.USERAGENT, CERN_SSO_CURL_USER_AGENT_KRB)
    curl.setopt(curl.HTTPAUTH, curl.HTTPAUTH_GSSNEGOTIATE)
    curl.setopt(curl.USERPWD, ':')

    # Follow location (and send the password along to other hosts,
    # although we do not really send any password)
    curl.setopt(curl.FOLLOWLOCATION, 1)
    curl.setopt(curl.UNRESTRICTED_AUTH, 1)

    # We do not need the headers
    curl.setopt(curl.HEADER, 0)

    # The target server has a valid certificate
    if secureTarget:
        curl.setopt(curl.SSL_VERIFYPEER, 1)
        curl.setopt(curl.SSL_VERIFYHOST, 2)
        curl.setopt(curl.CAPATH, CERN_SSO_CURL_CAPATH)
    else:
        curl.setopt(curl.SSL_VERIFYPEER, 0)
        curl.setopt(curl.SSL_VERIFYHOST, 0)

    # Fetch the url
    logging.debug('Connecting to %s', url)
    curl.setopt(curl.URL, url)
    (code, response, effectiveUrl) = perform()

    if CERN_SSO_CURL_ADFS_EP not in effectiveUrl:
        raise CERNSSOError('Not behind SSO or we already have the cookie.')

    # Do the manual redirection to the IDP
    logging.debug('Redirected to IDP %s', effectiveUrl)

    # The CERN SSO servers have a valid certificate
    if secure:
        curl.setopt(curl.SSL_VERIFYPEER, 1)
        curl.setopt(curl.SSL_VERIFYHOST, 2)
        curl.setopt(curl.CAPATH, CERN_SSO_CURL_CAPATH)
    else:
        curl.setopt(curl.SSL_VERIFYPEER, 0)
        curl.setopt(curl.SSL_VERIFYHOST, 0)

    curl.setopt(curl.URL, effectiveUrl)
    (code, response, effectiveUrl) = perform()

    if CERN_SSO_CURL_AUTHERR in response:
        raise CERNSSOError('Authentication error: Redirected to IDP Authentication error %s' % effectiveUrl)

    match = re.search('form .+?action="([^"]+)"', response)
    if not match:
        raise CERNSSOError('Something went wrong: could not find the expected redirection form (do you have a valid Kerberos ticket? -- see klist and kinit).')

    # Do the JavaScript redirection via the form to the SP
    spUrl = match.groups()[0]
    logging.debug('Redirected (via form) to SP %s', spUrl)

    formPairs = re.findall('input type="hidden" name="([^"]+)" value="([^"]+)"', response)

    # Microsoft ADFS produces broken encoding in auth forms:
    # '<' and '"' are encoded as '&lt;' and '&quot;' however
    # '>' is *not* encoded. Does not matter here though, we just decode.
    htmlParser = HTMLParser.HTMLParser()
    formPairs = [(x[0], htmlParser.unescape(x[1])) for x in formPairs]

    # The target server has a valid certificate
    if secureTarget:
        curl.setopt(curl.SSL_VERIFYPEER, 1)
        curl.setopt(curl.SSL_VERIFYHOST, 2)
        curl.setopt(curl.CAPATH, CERN_SSO_CURL_CAPATH)
    else:
        curl.setopt(curl.SSL_VERIFYPEER, 0)
        curl.setopt(curl.SSL_VERIFYHOST, 0)

    curl.setopt(curl.URL, spUrl)
    curl.setopt(curl.POSTFIELDS, urllib.urlencode(formPairs))
    curl.setopt(curl.POST, 1)
    (code, response, effectiveUrl) = perform()

    if CERN_SSO_CURL_ADFS_SIGNIN in effectiveUrl:
        raise CERNSSOError('Something went wrong: still on the auth page.')

    # Return the cookies
    return curl.getinfo(curl.INFO_COOKIELIST)


class HTTPError(Exception):
    '''A common HTTP exception.

    self.code is the response HTTP code as an integer.
    self.response is the response body (i.e. page).
    '''

    def __init__(self, code, response):
        self.code = code
        self.response = response

        # Try to extract the error message if possible (i.e. known error page format)
        try:
            self.args = (response.split('<p>')[1].split('</p>')[0], )
        except Exception:
            self.args = (self.response, )


class HTTP(object):
    '''Class used for querying URLs using the HTTP protocol.
    '''

    retryCodes = frozenset([502, 503])


    def __init__(self):
        self.setBaseUrl()
        self.setRetries()

        self.curl = pycurl.Curl()
        self.curl.setopt(self.curl.COOKIEFILE, '')
        self.curl.setopt(self.curl.SSL_VERIFYPEER, 0)
        self.curl.setopt(self.curl.SSL_VERIFYHOST, 0)


    def getCookies(self):
        '''Returns the list of cookies.
        '''

        return self.curl.getinfo(self.curl.INFO_COOKIELIST)


    def discardCookies(self):
        '''Discards cookies.
        '''

        self.curl.setopt(self.curl.COOKIELIST, 'ALL')


    def setBaseUrl(self, baseUrl = ''):
        '''Allows to set a base URL which will be prefixed to all the URLs
        that will be queried later.
        '''

        self.baseUrl = baseUrl


    def setProxy(self, proxy = ''):
        '''Allows to set a proxy.
        '''

        self.curl.setopt(self.curl.PROXY, proxy)


    def setTimeout(self, timeout = 0):
        '''Allows to set a timeout.
        '''

        self.curl.setopt(self.curl.TIMEOUT, timeout)


    def setRetries(self, retries = ()):
        '''Allows to set retries.

        The retries are a sequence of the seconds to wait per retry.

        The retries are done on:
            * PyCurl errors (includes network problems, e.g. not being able
              to connect to the host).
            * 502 Bad Gateway (for the moment, to avoid temporary
              Apache-CherryPy issues).
            * 503 Service Temporarily Unavailable (for when we update
              the frontends).
        '''

        self.retries = retries


    def query(self, url, data = None, files = None, keepCookies = True):
        '''Queries a URL, optionally with some data (dictionary).

        If no data is specified, a GET request will be used.
        If some data is specified, a POST request will be used.

        If files is specified, it must be a dictionary like data but
        the values are filenames.

        By default, cookies are kept in-between requests.

        A HTTPError exception is raised if the response's HTTP code is not 200.
        '''

        if not keepCookies:
            self.discardCookies()

        url = self.baseUrl + url

        # make sure the logs are safe ... at least somewhat :)
        data4log = copy.copy(data)
        if data4log:
            if 'password' in data4log.keys():
                data4log['password'] = '*'

        retries = [0] + list(self.retries)

        while True:
            logging.debug('Querying %s with data %s and files %s (retries left: %s, current sleep: %s)...', url, data4log, files, len(retries), retries[0])

            time.sleep(retries.pop(0))

            try:
                self.curl.setopt(self.curl.URL, url)
                self.curl.setopt(self.curl.HTTPGET, 1)

                if data is not None or files is not None:
                    # If there is data or files to send, use a POST request

                    finalData = {}

                    if data is not None:
                        finalData.update(data)

                    if files is not None:
                        for (key, fileName) in files.items():
                            finalData[key] = (self.curl.FORM_FILE, fileName)

                    self.curl.setopt(self.curl.HTTPPOST, finalData.items())

                response = cStringIO.StringIO()
                self.curl.setopt(self.curl.WRITEFUNCTION, response.write)
                self.curl.perform()

                code = self.curl.getinfo(self.curl.RESPONSE_CODE)

                if code in self.retryCodes and len(retries) > 0:
                    logging.debug('Retrying since we got the %s error code...', code)
                    continue

                if code != 200:
                    raise HTTPError(code, response.getvalue())

                return response.getvalue()

            except pycurl.error as e:
                if len(retries) == 0:
                    raise e

                logging.debug('Retrying since we got the %s pycurl exception...', str(e))


    def addCERNSSOCookies(self, url, secureTarget = True, secure = True):
        '''Adds the required CERN SSO cookies for a URL using Kerberos.

        After calling this, you can use query() for your SSO-protected URLs.

        This method will use your Kerberos ticket to sign in automatically
        in CERN SSO (i.e. no password required).

        If you do not have a ticket yet, use kinit.

        If you wish to make secure SSL connections to the CERN SSO
        (i.e. verify peers/hosts), you may need to install the CERN-CA-certs package.
        Use secure == False to skip this (i.e. this is the same as curl
        -k/--insecure). Not recommended: tell users to install them or use lxplus6.

        The same way, if you have a self-signed certificate in your target URL
        you can use secureTarget == False as well. Note that this option
        is provided in order to be able to use a secure SSL connection to CERN SSO,
        even if the connection to your target URL is not secure. Note that
        you will probably need the CERN-CA-certs package after you get a certificate
        signed by the CERN CA (https://cern.ch/ca), even if you did not need it
        for the CERN SSO.

        Note that this method *does* a query to the given URL if successful.

        Note that you may need different cookies for different URLs/applications.

        Note that this method may raise also CERNSSOError exceptions.
        '''

        for cookie in _getCERNSSOCookies(self.baseUrl + url, secureTarget, secure):
            self.curl.setopt(self.curl.COOKIELIST, cookie)

# common/http.py end


def addToTarFile(tarFile, fileobj, arcname):
    tarInfo = tarFile.gettarinfo(fileobj = fileobj, arcname = arcname)
    tarInfo.mode = 0400
    tarInfo.uid = tarInfo.gid = tarInfo.mtime = 0
    tarInfo.uname = tarInfo.gname = 'root'
    tarFile.addfile(tarInfo, fileobj)


class DropBox(object):
    '''A dropBox API class.
    '''

    def __init__(self, hostname = defaultHostname, urlTemplate = defaultUrlTemplate):
        self.hostname = hostname
        self.http = HTTP()
        self.http.setBaseUrl(urlTemplate % hostname)
        os.environ['http_proxy'] = 'http://cmsproxy.cms:3128/'
        os.environ['https_proxy'] = 'https://cmsproxy.cms:3128/'

    def signInSSO(self, secure = True):
        '''Signs in the server via CERN SSO.
        '''

        if secure:
            logging.info('%s: Signing in via CERN SSO...', self.hostname)
        else:
            logging.info('%s: Signing in via CERN SSO (insecure)...', self.hostname)

        # FIXME: Insecure connection to -prod until the certificates are fixed.
        #        The connection to the CERN SSO is still secure by default.
        #        On -dev and -int the certificates are installed properly.
        secureTarget = True
        if 'cms-conddb-prod' in self.hostname:
            secureTarget = False

        # We also use the CERN CA certificate to verify the targets,
        # so if we are not connecting securely to CERN SSO is because
        # we do not have the CERN-CA-certs package, so we need to skip
        # this as well.
        #
        # i.e. right now we have these options:
        #  secure == True,  secureTarget == True   with CERN CA cert, -dev and -int
        #  secure == True,  secureTarget == False  with CERN CA cert, -prod
        #  secure == False, secureTarget == False  without CERN CA cert
        if not secure:
            secureTarget = False

        self.http.addCERNSSOCookies('signInSSO', secureTarget, secure)


    def signIn(self, username, password):
        '''Signs in the server.
        '''

        logging.info('%s: Signing in...', self.hostname)
        self.http.query('signIn', {
            'username': username,
            'password': password,
        })


    def signOut(self):
        '''Signs out the server.
        '''

        logging.info('%s: Signing out...', self.hostname)
        self.http.query('signOut')


    def _checkForUpdates(self):
        '''Updates this script, if a new version is found.
        '''

        logging.info('%s: Checking for updates...', self.hostname)
        version = int(self.http.query('getUploadScriptVersion'))

        if version <= __version__:
            logging.info('%s: Up to date.', self.hostname)
            return

        logging.info('%s: There is a newer version (%s) than the current one (%s): Updating...', self.hostname, version, __version__)

        logging.info('%s: Downloading new version...', self.hostname)
        uploadScript = self.http.query('getUploadScript')

        self.signOut()

        logging.info('%s: Saving new version...', self.hostname)
        with open(sys.argv[0], 'wb') as f:
            f.write(uploadScript)

        logging.info('%s: Executing new version...', self.hostname)
        os.execl(sys.executable, *([sys.executable] + sys.argv))


    def uploadFile(self, filename, backend = defaultBackend, temporaryFile = defaultTemporaryFile):
        '''Uploads a file to the dropBox.

        The filename can be without extension, with .db or with .txt extension.
        It will be stripped and then both .db and .txt files are used.
        '''

        basepath = filename.rsplit('.db', 1)[0].rsplit('.txt', 1)[0]
        basename = os.path.basename(basepath)

        logging.info('%s: %s: Creating tar file...', self.hostname, basename)

        tarFile = tarfile.open(temporaryFile, 'w:bz2')

        with open('%s.db' % basepath, 'rb') as data:
            addToTarFile(tarFile, data, 'data.db')

        with tempfile.NamedTemporaryFile() as metadata:
            with open('%s.txt' % basepath, 'rb') as originalMetadata:
                json.dump(json.load(originalMetadata), metadata, sort_keys = True, indent = 4)

            metadata.seek(0)
            addToTarFile(tarFile, metadata, 'metadata.txt')

        tarFile.close()

        logging.info('%s: %s: Calculating hash...', self.hostname, basename)

        fileHash = hashlib.sha1()
        with open(temporaryFile, 'rb') as f:
            while True:
                data = f.read(4 * 1024 * 1024)

                if not data:
                    break

                fileHash.update(data)

        fileHash = fileHash.hexdigest()

        logging.info('%s: %s: Hash: %s', self.hostname, basename, fileHash)

        logging.info('%s: %s: Uploading file for the %s backend...', self.hostname, basename, backend)
        os.rename(temporaryFile, fileHash)
        self.http.query('uploadPopcon', {
            'backend': backend,
            'fileName': basename,
        }, files = {
            'uploadedFile': fileHash,
        })
        os.unlink(fileHash)


def getInput(default, prompt = ''):
    '''Like raw_input() but with a default and automatic strip().
    '''

    answer = raw_input(prompt)
    if answer:
        return answer.strip()

    return default.strip()


def getInputWorkflow(prompt = ''):
    '''Like getInput() but tailored to get target workflows (synchronization options).
    '''

    while True:
        workflow = getInput(defaultWorkflow, prompt)

        if workflow in frozenset(['offline', 'hlt', 'express', 'prompt', 'pcl']):
            return workflow

        logging.error('Please specify one of the allowed workflows. See above for the explanation on each of them.')


def getInputChoose(optionsList, default, prompt = ''):
    '''Makes the user choose from a list of options.
    '''

    while True:
        index = getInput(default, prompt)

        try:
            return optionsList[int(index)]
        except ValueError:
            logging.error('Please specify an index of the list (i.e. integer).')
        except IndexError:
            logging.error('The index you provided is not in the given list.')


def getInputRepeat(prompt = ''):
    '''Like raw_input() but repeats if nothing is provided and automatic strip().
    '''

    while True:
        answer = raw_input(prompt)
        if answer:
            return answer.strip()

        logging.error('You need to provide a value.')




