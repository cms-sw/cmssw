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
__version__ = 1


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


def runWizard(basename, dataFilename, metadataFilename):
    while True:
        print '''\nWizard for metadata for %s

I will ask you some questions to fill the metadata file. For some of the questions there are defaults between square brackets (i.e. []), leave empty (i.e. hit Enter) to use them.''' % basename

        # Try to get the available inputTags
        try:
            dataConnection = sqlite3.connect(dataFilename)
            dataCursor = dataConnection.cursor()
            dataCursor.execute('select name from sqlite_master where type == "table"')
            tables = set(zip(*dataCursor.fetchall())[0])

            # only conddb V2 supported...
            if 'TAG' in tables:
                dataCursor.execute('select NAME from TAG')
            # In any other case, do not try to get the inputTags
            else:
                raise Exception()

            inputTags = dataCursor.fetchall()
            if len(inputTags) == 0:
                raise Exception()
            inputTags = zip(*inputTags)[0]

        except Exception:
            inputTags = []

        if len(inputTags) == 0:
            print '\nI could not find any input tag in your data file, but you can still specify one manually.'

            inputTag = getInputRepeat(
                '\nWhich is the input tag (i.e. the tag to be read from the SQLite data file)?\ne.g. BeamSpotObject_ByRun\ninputTag: ')

        else:
            print '\nI found the following input tags in your SQLite data file:'
            for (index, inputTag) in enumerate(inputTags):
                print '   %s) %s' % (index, inputTag)

            inputTag = getInputChoose(inputTags, '0',
                                      '\nWhich is the input tag (i.e. the tag to be read from the SQLite data file)?\ne.g. 0 (you select the first in the list)\ninputTag [0]: ')

        destinationDatabase = ''
        ntry = 0
        while ( destinationDatabase != 'oracle://cms_orcon_prod/CMS_CONDITIONS' and destinationDatabase != 'oracle://cms_orcoff_prep/CMS_CONDITIONS' ): 
            if ntry==0:
                inputMessage = \
                '\nWhich is the destination database where the tags should be exported? \nPossible choices: oracle://cms_orcon_prod/CMS_CONDITIONS (for prod) or oracle://cms_orcoff_prep/CMS_CONDITIONS (for prep) \ndestinationDatabase: '
            elif ntry==1:
                inputMessage = \
                '\nPlease choose one of the two valid destinations: \noracle://cms_orcon_prod/CMS_CONDITIONS (for prod) or oracle://cms_orcoff_prep/CMS_CONDITIONS (for prep) \
\ndestinationDatabase: '
            else:
                raise Exception('No valid destination chosen. Bailing out...')
            destinationDatabase = getInputRepeat(inputMessage)
            ntry += 1

        while True:
            since = getInput('',
                             '\nWhich is the given since? (if not specified, the one from the SQLite data file will be taken -- note that even if specified, still this may not be the final since, depending on the synchronization options you select later: if the synchronization target is not offline, and the since you give is smaller than the next possible one (i.e. you give a run number earlier than the one which will be started/processed next in prompt/hlt/express), the DropBox will move the since ahead to go to the first safe run instead of the value you gave)\ne.g. 1234\nsince []: ')
            if not since:
                since = None
                break
            else:
                try:
                    since = int(since)
                    break
                except ValueError:
                    logging.error('The since value has to be an integer or empty (null).')

        userText = getInput('',
                            '\nWrite any comments/text you may want to describe your request\ne.g. Muon alignment scenario for...\nuserText []: ')

        destinationTags = {}
        while True:
            destinationTag = getInput('',
                                      '\nWhich is the next destination tag to be added (leave empty to stop)?\ne.g. BeamSpotObjects_PCL_byRun_v0_offline\ndestinationTag []: ')
            if not destinationTag:
                if len(destinationTags) == 0:
                    logging.error('There must be at least one destination tag.')
                    continue
                break

            if destinationTag in destinationTags:
                logging.warning(
                    'You already added this destination tag. Overwriting the previous one with this new one.')

            destinationTags[destinationTag] = {
            }

        metadata = {
            'destinationDatabase': destinationDatabase,
            'destinationTags': destinationTags,
            'inputTag': inputTag,
            'since': since,
            'userText': userText,
        }

        metadata = json.dumps(metadata, sort_keys=True, indent=4)
        print '\nThis is the generated metadata:\n%s' % metadata

        if getInput('n',
                    '\nIs it fine (i.e. save in %s and *upload* the conditions if this is the latest file)?\nAnswer [n]: ' % metadataFilename).lower() == 'y':
            break
    logging.info('Saving generated metadata in %s...', metadataFilename)
    with open(metadataFilename, 'wb') as metadataFile:
        metadataFile.write(metadata)

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


    def _checkForUpdates(self):
        '''Updates this script, if a new version is found.
        '''

        logging.debug('%s: Checking if a newer version of this script is available ...', self.hostname)
        version = int(self.http.query('getUploadScriptVersion'))

        if version <= __version__:
            logging.debug('%s: Script is up-to-date.', self.hostname)
            return

        logging.info('%s: Updating to a newer version (%s) than the current one (%s): downloading ...', self.hostname, version, __version__)

        uploadScript = self.http.query('getUploadScript')

        self.signOut()

        logging.info('%s: ... saving the new version ...', self.hostname)
        with open(sys.argv[0], 'wb') as f:
            f.write(uploadScript)

        logging.info('%s: ... executing the new version...', self.hostname)
        os.execl(sys.executable, *([sys.executable] + sys.argv))


    def uploadFile(self, filename, backend = defaultBackend, temporaryFile = defaultTemporaryFile):
        '''Uploads a file to the dropBox.

        The filename can be without extension, with .db or with .txt extension.
        It will be stripped and then both .db and .txt files are used.
        '''

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

def authenticateUser(dropBox, options):

    try:
        # Try to find the netrc entry
        (username, account, password) = netrc.netrc().authenticators(options.netrcHost)
    except Exception:
        # netrc entry not found, ask for the username and password
        logging.info(
            'netrc entry "%s" not found: if you wish not to have to retype your password, you can add an entry in your .netrc file. However, beware of the risks of having your password stored as plaintext. Instead.',
            options.netrcHost)

        # Try to get a default username
        defaultUsername = getpass.getuser()
        if defaultUsername is None:
            defaultUsername = '(not found)'

        username = getInput(defaultUsername, '\nUsername [%s]: ' % defaultUsername)
        password = getpass.getpass('Password: ')

    # Now we have a username and password, authenticate with them
    return dropBox.signIn(username, password)


def uploadAllFiles(options, arguments):
    
    ret = {}
    ret['status'] = 0

    # Check that we can read the data and metadata files
    # If the metadata file does not exist, start the wizard
    for filename in arguments:
        basepath = filename.rsplit('.db', 1)[0].rsplit('.txt', 1)[0]
        basename = os.path.basename(basepath)
        dataFilename = '%s.db' % basepath
        metadataFilename = '%s.txt' % basepath

        logging.info('Checking %s...', basename)

        # Data file
        try:
            with open(dataFilename, 'rb') as dataFile:
                pass
        except IOError as e:
            errMsg = 'Impossible to open SQLite data file %s' %dataFilename
            logging.error( errMsg )
            ret['status'] = -3
            ret['error'] = errMsg
            return ret

        # Check the data file
        empty = True
        try:
            dbcon = sqlite3.connect( dataFilename )
            dbcur = dbcon.cursor()
            dbcur.execute('SELECT * FROM IOV')
            rows = dbcur.fetchall()
            for r in rows:
                empty = False
            dbcon.close()
            if empty:
                errMsg = 'The input SQLite data file %s contains no data.' %dataFilename
                logging.error( errMsg )
                ret['status'] = -4
                ret['error'] = errMsg
                return ret
        except Exception as e:
            errMsg = 'Check on input SQLite data file %s failed: %s' %(dataFilename,str(e))
            logging.error( errMsg )
            ret['status'] = -5
            ret['error'] = errMsg
            return ret

        # Metadata file
        try:
            with open(metadataFilename, 'rb') as metadataFile:
                pass
        except IOError as e:
            if e.errno != errno.ENOENT:
                errMsg = 'Impossible to open file %s (for other reason than not existing)' %metadataFilename
                logging.error( errMsg )
                ret['status'] = -4
                ret['error'] = errMsg
                return ret

            if getInput('y', '\nIt looks like the metadata file %s does not exist. Do you want me to create it and help you fill it?\nAnswer [y]: ' % metadataFilename).lower() != 'y':
                errMsg = 'Metadata file %s does not exist' %metadataFilename
                logging.error( errMsg )
                ret['status'] = -5
                ret['error'] = errMsg
                return ret
            # Wizard
            runWizard(basename, dataFilename, metadataFilename)

    # Upload files
    try:
        dropBox = ConditionsUploader(options.hostname, options.urlTemplate)

        # Authentication
        if not authenticateUser(dropBox, options):
            logging.error("Error authenticating user. Aborting.")
            return { 'status' : -2, 'error' : "Error authenticating user. Aborting." }

        # At this point we must be authenticated
        dropBox._checkForUpdates()

        results = {}
        for filename in arguments:
            backend = options.backend
            basepath = filename.rsplit('.db', 1)[0].rsplit('.txt', 1)[0]
            metadataFilename = '%s.txt' % basepath
            with open(metadataFilename, 'rb') as metadataFile:
                metadata = json.load( metadataFile )
            # When dest db = prep the hostname has to be set to dev.
            forceHost = False
            destDb = metadata['destinationDatabase']
            if destDb.startswith('oracle://cms_orcon_prod') or destDb.startswith('oracle://cms_orcoff_prep'):
                if destDb.startswith('oracle://cms_orcoff_prep'):
                    dropBox.setHost( defaultDevHostname )
                    dropBox.signInAgain()
                    forceHost = True
                results[filename] = dropBox.uploadFile(filename, options.backend, options.temporaryFile)
                if forceHost:
                    # set back the hostname to the original global setting
                    dropBox.setHost( options.hostname )
                    dropBox.signInAgain()
            else:
                results[filename] = False
                logging.error("DestinationDatabase %s is not valid. Skipping the upload." %destDb)
            if not results[filename]:
                if ret['status']<0:
                    ret['status'] = 0
                ret['status'] += 1
        ret['files'] = results
        logging.debug("all files processed, logging out now.")

        dropBox.signOut()

    except HTTPError as e:
        logging.error('got HTTP error: %s', str(e))
        return { 'status' : -1, 'error' : str(e) }

    return ret

def uploadTier0Files(filenames, username, password, cookieFileName = None):
    '''Uploads a bunch of files coming from Tier0.
    This has the following requirements:
        * Username/Password based authentication.
        * Uses the online backend.
        * Ignores errors related to the upload/content (e.g. duplicated file).
    '''

    dropBox = ConditionsUploader()

    dropBox.signIn(username, password)

    for filename in filenames:
        try:
            result = dropBox.uploadFile(filename, backend = 'test')
        except HTTPError as e:
            if e.code == 400:
                # 400 Bad Request: This is an exception related to the upload
                # being wrong for some reason (e.g. duplicated file).
                # Since for Tier0 this is not an issue, continue
                logging.error('HTTP Exception 400 Bad Request: Upload-related, skipping. Message: %s', e)
                continue

            # In any other case, re-raise.
            raise

        #-toDo: add a flag to say if we should retry or not. So far, all retries are done server-side (Tier-0),
        #       if we flag as failed any retry would not help and would result in the same error (e.g.
        #       when a file with an identical hash is uploaded again)
        #-review(2015-09-25): get feedback from tests at Tier-0 (action: AP)

        if not result: # dropbox reported an error when uploading, do not retry.
            logging.error('Error from dropbox, upload-related, skipping.')
            continue

    dropBox.signOut()


def main():
    '''Entry point.
    '''

    parser = optparse.OptionParser(usage =
        'Usage: %prog [options] <file> [<file> ...]\n'
    )

    parser.add_option('-d', '--debug',
        dest = 'debug',
        action="store_true",
        default = False,
        help = 'Switch on printing debug information. Default: %default',
    )

    parser.add_option('-b', '--backend',
        dest = 'backend',
        default = defaultBackend,
        help = 'dropBox\'s backend to upload to. Default: %default',
    )

    parser.add_option('-H', '--hostname',
        dest = 'hostname',
        default = defaultHostname,
        help = 'dropBox\'s hostname. Default: %default',
    )

    parser.add_option('-u', '--urlTemplate',
        dest = 'urlTemplate',
        default = defaultUrlTemplate,
        help = 'dropBox\'s URL template. Default: %default',
    )

    parser.add_option('-f', '--temporaryFile',
        dest = 'temporaryFile',
        default = defaultTemporaryFile,
        help = 'Temporary file that will be used to store the first tar file. Note that it then will be moved to a file with the hash of the file as its name, so there will be two temporary files created in fact. Default: %default',
    )

    parser.add_option('-n', '--netrcHost',
        dest = 'netrcHost',
        default = defaultNetrcHost,
        help = 'The netrc host (machine) from where the username and password will be read. Default: %default',
    )

    (options, arguments) = parser.parse_args()

    if len(arguments) < 1:
        parser.print_help()
        return -2

    logLevel = logging.INFO
    if options.debug:
        logLevel = logging.DEBUG
    logging.basicConfig(
        format = '[%(asctime)s] %(levelname)s: %(message)s',
        level = logLevel,
    )

    results = uploadAllFiles(options, arguments)

    if not results.has_key('status'):
        print 'Unexpected error.'
        return -1
    ret = results['status']
    print results
    print "upload ended with code: %s" %ret
    #for hash, res in results.items():
    #    print "\t %s : %s " % (hash, str(res))
    return ret

def testTier0Upload():

    global defaultNetrcHost

    (username, account, password) = netrc.netrc().authenticators(defaultNetrcHost)

    filenames = ['testFiles/localSqlite-top2']

    uploadTier0Files(filenames, username, password, cookieFileName = None)


if __name__ == '__main__':

    sys.exit(main())
    # testTier0Upload()
