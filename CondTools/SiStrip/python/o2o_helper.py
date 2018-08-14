'''
Helper Script for StripO2O
@author: Huilin Qu
'''

import os
import subprocess
import logging
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import sqlite3
import six

def kill_subproc_noexcept(p):
    '''Kill a subprocess without throwing OSError.
    Used for cleaning up subprocesses when the main script crashes.'''
    try:
        p.terminate()
    except OSError:
        pass


def configLogger(logfile,loglevel=logging.INFO):
    '''Setting up logging to both file and console.
    @see: https://docs.python.org/2/howto/logging-cookbook.html
    '''
    # set up logging to file
    logging.basicConfig(level=loglevel,
                        format='[%(asctime)s] %(levelname)s: %(message)s',
                        filename=logfile,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(loglevel)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

def insert_to_file(template, target, replace_dict):
    '''Update the template file based on the replace_dict, and write to the target.'''
    logging.debug('Creating "%s" from template "%s" using dictionary:'%(target, template))
    logging.debug(replace_dict)
    with open(template, 'r') as input_file:
        config=input_file.read()
    with open(target, 'w') as output_file:
        for key, value in six.iteritems(replace_dict):
            config = config.replace(key, value)
        output_file.write(config)
    return config

def create_metadata(metadataFilename, inputTag, destTags, destDb, since, userText):
    '''Create metadata file for the conditionsUpload service.
    @see: uploadConditions.runWizard()
    @see: https://twiki.cern.ch/twiki/bin/view/CMS/DropBox

    Keyword arguments:
    metadataFilename -- output metadata filename
    inputTag -- input tag name
    destTags -- a list of destination tags
    destDb -- [destinationDatabase] in metadata
    since -- [since] in metadata
    userText -- [userText] in metadata
    '''
    if isinstance(destTags, str):
        destTags = [destTags]
    if since:
        since = int(since)  # convert to int if since is not None (input since can be a str)
    destinationTags = {}
    for destinationTag in destTags:
        destinationTags[destinationTag] = {}
    metadata = {
        'destinationDatabase': destDb,
        'destinationTags': destinationTags,
        'inputTag': inputTag,
        'since': since,
        'userText': userText,
    }
    logging.info('Writing metadata in %s', metadataFilename)
    logging.debug(metadata)
    with open(metadataFilename, 'wb') as metadataFile:
        metadataFile.write(json.dumps(metadata, sort_keys=True, indent=4))

def upload_payload(dbFile, inputTag, destTags, destDb, since, userText):
    '''Upload payload using conditionUploader. '''
    if isinstance(destTags, str):
        destTags = [destTags]
    metadataFilename = dbFile.replace('.db', '.txt')
    create_metadata(metadataFilename, inputTag, destTags, destDb, since, userText)
    logging.info('Uploading tag [%s] from %s to [%s] in %s:' % (inputTag, dbFile, ','.join(destTags), destDb))
    command = "uploadConditions.py %s" % dbFile
    pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = pipe.communicate()[0]
    logging.info(out)
    logging.info('@@@Upload return code = %d@@@' % pipe.returncode)
    if pipe.returncode != 0:
        raise RuntimeError('Upload FAILED!')


def copy_payload(dbFile, inputTag, destTags, destDb, since, userText):
    '''Upload payload using conddb copy.'''
    if isinstance(destTags, str):
        destTags = [destTags]
    if destDb.lower() == 'oracle://cms_orcon_prod/cms_conditions':
        copyDestDb = 'onlineorapro'
    elif destDb.lower() == 'oracle://cms_orcoff_prep/cms_conditions':
        copyDestDb = 'oradev'
    else:
        copyDestDb = destDb
    success = 0
    def copy(dest):
        command = 'conddb --force --yes --db {db} copy {inputTag} {destTag} --destdb {destDb} --synchronize --note "{note}"'.format(
            db=dbFile, inputTag=inputTag, destTag=dest, destDb=copyDestDb, note=userText)
        logging.info('Copy tag [%s] from %s to [%s] in %s:' % (inputTag, dbFile, dest, destDb))
        logging.debug(command)
        pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = pipe.communicate()[0]
        logging.info(out)
        return pipe.returncode
    for dest in destTags:
        returncode = copy(dest)
        if returncode == 0: success += 1
    logging.info('@@@Upload return code = %d@@@' % (success - len(destTags)))
    if success != len(destTags):
        raise RuntimeError('Upload FAILED!')


def send_mail(subject, message, send_to, send_from, text_attachments=[]):
    '''Send an email. [send_to] needs to be a list.'''
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = send_from
    msg['To'] = ','.join(send_to)
    msg.attach(MIMEText(message))

    for fn in text_attachments:
        with open(fn, 'rb') as txtfile:
            attachment = MIMEText(txtfile.read())
            attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(fn))
            msg.attach(attachment)

    s = smtplib.SMTP('localhost')
    s.sendmail(send_from, send_to, msg.as_string())
    s.quit()

def exists_iov(dbFile, tag):
    '''Check if there exists any IOV for a specific tag in the given sqlite file.'''
    dataConnection = sqlite3.connect(dbFile)
    dataCursor = dataConnection.cursor()
    dataCursor.execute('select SINCE from IOV where TAG_NAME=:tag_name', {'tag_name' : tag})
    return len(dataCursor.fetchall()) > 0
