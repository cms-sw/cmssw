import subprocess
import json
import netrc
import sqlite3
import os
import sys
import shutil
import logging
from datetime import datetime

dbName = 'popcon'
dbFileName = '%s.db' %dbName
dbFileForDropBox = dbFileName
dbLogFile = '%s_log.db' %dbName
errorInImportFileFolder = 'import_errors'
dateformatForFolder = "%y-%m-%d-%H-%M-%S"
dateformatForLabel = "%y-%m-%d %H:%M:%S"

auth_path_key = 'COND_AUTH_PATH'

messageLevelEnvVar = 'POPCON_LOG_LEVEL'
fmt_str = "[%(asctime)s] %(levelname)s: %(message)s"
logLevel = logging.INFO
if messageLevelEnvVar in os.environ:
    levStr = os.environ[messageLevelEnvVar]
    if levStr == 'DEBUG':
        logLevel = logging.DEBUG
logFormatter = logging.Formatter(fmt_str)
logger = logging.getLogger()        
logger.setLevel(logLevel)
consoleHandler = logging.StreamHandler(sys.stdout) 
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

def checkFile():
    # check if the expected input file is there... 
    # exit code < 0 => error
    # exit code = 0 => skip
    # exit code = 1 => import                                                                                                                             
    if not os.path.exists( dbFileName ):
       logger.error('The file expected as an input %s has not been found.'%dbFileName )
       return -1

    empty = True
    try:
       dbcon = sqlite3.connect( dbFileName )
       dbcur = dbcon.cursor()
       dbcur.execute('SELECT * FROM IOV')
       rows = dbcur.fetchall()
       for r in rows:
          empty = False
       dbcon.close()
       if empty:
           logger.warning('The file expected as an input %s contains no data. The import will be skipped.'%dbFileName )
           return 0
       return 1
    except Exception as e:
       logger.error('Check on input data failed: %s' %str(e))
       return -2

def saveFileForImportErrors( datef, withMetadata=False ):
    # save a copy of the files in case of upload failure...
    leafFolderName = datef.strftime(dateformatForFolder)
    fileFolder = os.path.join( errorInImportFileFolder, leafFolderName)
    if not os.path.exists(fileFolder):
        os.makedirs(fileFolder)
    df= '%s.db' %dbName
    dataDestFile = os.path.join( fileFolder, df)
    if not os.path.exists(dataDestFile):
        shutil.copy2(df, dataDestFile)
    if withMetadata:
        mf= '%s.txt' %dbName
        metadataDestFile = os.path.join( fileFolder, mf )
        if not os.path.exists(metadataDestFile):
            shutil.copy2(df, metadataDestFile)
    logger.error("Upload failed. Data file and metadata saved in folder '%s'" %os.path.abspath(fileFolder))
    
def upload( args ):
    destDb = args.destDb
    destTag = args.destTag
    comment = args.comment

    datef = datetime.now()

    # first remove any existing metadata file...                                                                                                                                
    if os.path.exists( '%s.txt' %dbName ):
       logger.debug('Removing already existing file %s' %dbName)
       os.remove( '%s.txt' %dbName )
    
    # dump Metadata for the Upload
    uploadMd = {}
    uploadMd['destinationDatabase'] = destDb
    tags = {}
    tagInfo = {}
    tags[ destTag ] = tagInfo
    uploadMd['destinationTags'] = tags
    uploadMd['inputTag'] = destTag
    uploadMd['since'] = None
    datelabel = datef.strftime(dateformatForLabel)
    commentStr = ''
    if not comment is None:
       commentStr = comment
    uploadMd['userText'] = '%s : %s' %(datelabel,commentStr)
    with open( '%s.txt' %dbName, 'wb') as jf:
       jf.write( json.dumps( uploadMd, sort_keys=True, indent = 2 ) )
       jf.write('\n')

    # run the upload
    uploadCommand = 'uploadConditions.py %s' %dbName
    ret = 0
    try:
       pipe = subprocess.Popen( uploadCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
       stdout = pipe.communicate()[0]
       print stdout
       retCode = pipe.returncode
       if retCode != 0:
           saveFileForImportErrors( datef, True )
       ret |= retCode
    except Exception as e:
       ret |= 1
       logger.error(str(e))
    return ret

def copy( args ):
    destDb = args.destDb
    destTag = args.destTag
    comment = args.comment

    datef = datetime.now()
    destMap = { "oracle://cms_orcoff_prep/cms_conditions": "oradev", "oracle://cms_orcon_prod/cms_conditions": "onlineorapro"  }
    if destDb.lower() in destMap.keys():
        destDb = destMap[destDb.lower()]
    else:
        logger.error( 'Destination connection %s is not supported.' %destDb )
        return 
    # run the copy
    note = '"Importing data with O2O execution"'
    commandOptions = '--force --yes --db %s copy %s %s --destdb %s --synchronize --note %s' %(dbFileName,destTag,destTag,destDb,note)
    copyCommand = 'conddb %s' %commandOptions
    logger.info( 'Executing command: %s' %copyCommand )
    try:
        pipe = subprocess.Popen( copyCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
        stdout = pipe.communicate()[0]
        print stdout
        retCode = pipe.returncode
        if retCode != 0:
            saveFileForImportErrors( datef )
        ret = retCode
    except Exception as e:
        ret = 1
        logger.error( str(e) )
    return ret

def run( args ):
    if args.auth is not None and not args.auth=='':
        if auth_path_key in os.environ:
            logger.warning("Cannot set authentication path to %s in the environment, since it is already set." %args.auth)
        else:
            logger.info("Setting the authentication path to %s in the environment." %args.auth)
            os.environ[auth_path_key]=args.auth
    if os.path.exists( '%s.db' %dbName ):
       logger.info("Removing files with name %s" %dbName )
       os.remove( '%s.db' %dbName )
    if os.path.exists( '%s.txt' %dbName ):
       os.remove( '%s.txt' %dbName )
    command = 'cmsRun %s ' %args.job_file
    command += ' destinationDatabase=%s' %args.destDb
    command += ' destinationTag=%s' %args.destTag
    command += ' 2>&1'
    pipe = subprocess.Popen( command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
    stdout = pipe.communicate()[0]
    retCode = pipe.returncode
    print stdout
    logger.info('PopCon Analyzer return code is: %s' %retCode )
    if retCode!=0:
       logger.error( 'O2O job failed. Skipping upload.' )
       return retCode

    ret = checkFile()
    if ret < 0:
        return ret
    elif ret == 0:
        return 0
    if args.copy:
        return copy( args )
    return upload( args )
