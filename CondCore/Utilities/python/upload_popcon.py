#!/usr/bin/env python
'''Script that uploads to the new CMS conditions uploader.
Adapted to the new infrastructure from v6 of the upload.py script for the DropBox from Miguel Ojeda.
'''

__author__ = 'Andreas Pfeiffer'
__copyright__ = 'Copyright 2015, CERN CMS'
__credits__ = ['Giacomo Govi', 'Salvatore Di Guida', 'Miguel Ojeda', 'Andreas Pfeiffer']
__license__ = 'Unknown'
__maintainer__ = 'Andreas Pfeiffer'
__email__ = 'andreas.pfeiffer@cern.ch'


import os
import sys
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
defaultDevHostname = 'cms-conddb-dev.cern.ch'
defaultUrlTemplate = 'https://%s/cmsDbUpload/'
defaultTemporaryFile = 'upload.tar.bz2'
defaultNetrcHost = 'ConditionUploader'
defaultWorkflow = 'offline'

# common/http.py start (plus the "# Try to extract..." section bit)
import time
import logging
import cStringIO

import pycurl
import socket
import copy


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
            

CERN_SSO_CURL_CAPATH = '/etc/pki/tls/certs'

class HTTP(object):
    '''Class used for querying URLs using the HTTP protocol.
    '''

    retryCodes = frozenset([502, 503])

    def __init__(self):
        self.setBaseUrl()
        self.setRetries()

        self.curl = pycurl.Curl()
        self.curl.setopt(self.curl.COOKIEFILE, '')      # in memory

        #-toDo: make sure we have the right options set here to use ssl
        #-review(2015-09-25): check and see - action: AP
        # self.curl.setopt(self.curl.SSL_VERIFYPEER, 1)
        self.curl.setopt(self.curl.SSL_VERIFYPEER, 0)
        self.curl.setopt(self.curl.SSL_VERIFYHOST, 2)

        self.baseUrl = None

        self.token = None

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

    def getToken(self, username, password):

        url = self.baseUrl + 'token'

        self.curl.setopt(pycurl.URL, url)
        self.curl.setopt(pycurl.VERBOSE, 0)

        #-toDo: check if/why these are needed ...
        #-ap: hmm ...
        # self.curl.setopt(pycurl.DNS_CACHE_TIMEOUT, 0)
        # self.curl.setopt(pycurl.IPRESOLVE, pycurl.IPRESOLVE_V4)
        #-end hmmm ...
        #-review(2015-09-25): check and see - action: AP


        self.curl.setopt(pycurl.HTTPHEADER, ['Accept: application/json'])
        # self.curl.setopt( self.curl.POST, {})
        self.curl.setopt(self.curl.HTTPGET, 0)

        response = cStringIO.StringIO()
        self.curl.setopt(pycurl.WRITEFUNCTION, response.write)
        self.curl.setopt(pycurl.USERPWD, '%s:%s' % (username, password) )

        logging.debug('going to connect to server at: %s' % url )

        self.curl.perform()
        code = self.curl.getinfo(pycurl.RESPONSE_CODE)
        logging.debug('got: %s ', str(code))
        
        try:
            self.token = json.loads( response.getvalue() )['token']
        except Exception as e:
            logging.error('http::getToken> got error from server: %s ', str(e) )
            if 'No JSON object could be decoded' in str(e):
                return None
            logging.error("error getting token: %s", str(e))
            return None

        logging.debug('token: %s', self.token)
        logging.debug('returning: %s', response.getvalue())

        return response.getvalue()

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

                # from now on we use the token we got from the login
                self.curl.setopt(pycurl.USERPWD, '%s:""' % ( str(self.token), ) )
                self.curl.setopt(pycurl.HTTPHEADER, ['Accept: application/json'])

                if data is not None or files is not None:
                    # If there is data or files to send, use a POST request

                    finalData = {}

                    if data is not None:
                        finalData.update(data)

                    if files is not None:
                        for (key, fileName) in files.items():
                            finalData[key] = (self.curl.FORM_FILE, fileName)
                    self.curl.setopt( self.curl.HTTPPOST, finalData.items() )

                self.curl.setopt(pycurl.VERBOSE, 0)

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

# common/http.py end

def addToTarFile(tarFile, fileobj, arcname):
    tarInfo = tarFile.gettarinfo(fileobj = fileobj, arcname = arcname)
    tarInfo.mode = 0o400
    tarInfo.uid = tarInfo.gid = tarInfo.mtime = 0
    tarInfo.uname = tarInfo.gname = 'root'
    tarFile.addfile(tarInfo, fileobj)

class ConditionsUploader(object):
    '''Upload conditions to the CMS conditions uploader service.
    '''

    def __init__(self, hostname = defaultHostname, urlTemplate = defaultUrlTemplate):
        self.hostname = hostname
        self.urlTemplate = urlTemplate 
        self.userName = None
        self.http = None
        self.password = None

    def setHost( self, hostname ):
        self.hostname = hostname

    def signIn(self, username, password):
        ''' init the server.
        '''
        self.http = HTTP()
        if socket.getfqdn().strip().endswith('.cms'):
            self.http.setProxy('https://cmsproxy.cms:3128/')
        self.http.setBaseUrl(self.urlTemplate % self.hostname)
        '''Signs in the server.
        '''

        logging.info('%s: Signing in user %s ...', self.hostname, username)
        try:
            self.token = self.http.getToken(username, password)
        except Exception as e:
            logging.error("Caught exception when trying to get token for user %s from %s: %s" % (username, self.hostname, str(e)) )
            return False

        if not self.token:
            logging.error("could not get token for user %s from %s" % (username, self.hostname) )
            return False

        logging.debug( "got: '%s'", str(self.token) )
        self.userName = username
        self.password = password
        return True

    def signInAgain(self):
        return self.signIn( self.userName, self.password )

    def signOut(self):
        '''Signs out the server.
        '''

        logging.info('%s: Signing out...', self.hostname)
        # self.http.query('logout')
        self.token = None


    def uploadFile(self, filename, backend = defaultBackend, temporaryFile = defaultTemporaryFile):
        '''Uploads a file to the dropBox.

        The filename can be without extension, with .db or with .txt extension.
        It will be stripped and then both .db and .txt files are used.
        '''

        basepath = filename.rsplit('.db', 1)[0].rsplit('.txt', 1)[0]
        metadataFilename = '%s.txt' % basepath
        with open(metadataFilename, 'rb') as metadataFile:
            metadata = json.load( metadataFile )
        # When dest db = prep the hostname has to be set to dev.
        forceHost = False
        destDb = metadata['destinationDatabase']
        ret = False
        if destDb.startswith('oracle://cms_orcon_prod') or destDb.startswith('oracle://cms_orcoff_prep'):
            if destDb.startswith('oracle://cms_orcoff_prep'):
                    self.setHost( defaultDevHostname )
                    self.signInAgain()
                    forceHost = True
            ret = self._uploadFile(filename, backend, temporaryFile)
            if forceHost:
                # set back the hostname to the original global setting
                self.setHost( defaultHostname )
                self.signInAgain()
        else:
            logging.error("DestinationDatabase %s is not valid. Skipping the upload." %destDb)
        return ret

    def _uploadFile(self, filename, backend = defaultBackend, temporaryFile = defaultTemporaryFile):

        basepath = filename.rsplit('.db', 1)[0].rsplit('.txt', 1)[0]
        basename = os.path.basename(basepath)

        logging.debug('%s: %s: Creating tar file for upload ...', self.hostname, basename)

        try:
            tarFile = tarfile.open(temporaryFile, 'w:bz2')

            with open('%s.db' % basepath, 'rb') as data:
                addToTarFile(tarFile, data, 'data.db')
        except Exception as e:
            msg = 'Error when creating tar file. \n'
            msg += 'Please check that you have write access to the directory you are running,\n'
            msg += 'and that you have enough space on this disk (df -h .)\n'
            logging.error(msg)
            raise Exception(msg)

        with tempfile.NamedTemporaryFile() as metadata:
            with open('%s.txt' % basepath, 'rb') as originalMetadata:
                json.dump(json.load(originalMetadata), metadata, sort_keys = True, indent = 4)

            metadata.seek(0)
            addToTarFile(tarFile, metadata, 'metadata.txt')

        tarFile.close()

        logging.debug('%s: %s: Calculating hash...', self.hostname, basename)

        fileHash = hashlib.sha1()
        with open(temporaryFile, 'rb') as f:
            while True:
                data = f.read(4 * 1024 * 1024)
                if not data:
                    break
                fileHash.update(data)

        fileHash = fileHash.hexdigest()
        fileInfo = os.stat(temporaryFile)
        fileSize = fileInfo.st_size

        logging.debug('%s: %s: Hash: %s', self.hostname, basename, fileHash)

        logging.info('%s: %s: Uploading file (%s, size %s) to the %s backend...', self.hostname, basename, fileHash, fileSize, backend)
        os.rename(temporaryFile, fileHash)
        try:
            ret = self.http.query('uploadFile',
                              {
                                'backend': backend,
                                'fileName': basename,
                                'userName': self.userName,
                              },
                              files = {
                                        'uploadedFile': fileHash,
                                      }
                              )
        except Exception as e:
            logging.error('Error from uploading: %s' % str(e))
            ret = json.dumps( { "status": -1, "upload" : { 'itemStatus' : { basename : {'status':'failed', 'info':str(e)}}}, "error" : str(e)} )

        os.unlink(fileHash)

        statusInfo = json.loads(ret)['upload']
        logging.debug( 'upload returned: %s', statusInfo )

        okTags      = []
        skippedTags = []
        failedTags  = []
        for tag, info in statusInfo['itemStatus'].items():
            logging.debug('checking tag %s, info %s', tag, str(json.dumps(info, indent=4,sort_keys=True)) )
            if 'ok'   in info['status'].lower() :
                okTags.append( tag )
                logging.info('tag %s successfully uploaded', tag)
            if 'skip' in info['status'].lower() :
                skippedTags.append( tag )
                logging.warning('found tag %s to be skipped. reason:  \n ... \t%s ', tag, info['info'])
            if 'fail' in info['status'].lower() :
                failedTags.append( tag )
                logging.error('found tag %s failed to upload. reason: \n ... \t%s ', tag, info['info'])

        if len(okTags)      > 0: logging.info   ("tags sucessfully uploaded: %s ", str(okTags) )
        if len(skippedTags) > 0: logging.warning("tags SKIPped to upload   : %s ", str(skippedTags) )
        if len(failedTags)  > 0: logging.error  ("tags FAILed  to upload   : %s ", str(failedTags) )

        fileLogURL = 'https://%s/logs/dropBox/getFileLog?fileHash=%s' 
        logging.info('file log at: %s', fileLogURL % (self.hostname,fileHash))

        return len(okTags)>0
