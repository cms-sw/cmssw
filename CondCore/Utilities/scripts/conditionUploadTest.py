#!/usr/bin/env python

import cx_Oracle
import subprocess
import json
import os
import shutil
import datetime

# Requirement 1: a conddb key for the authentication with valid permission on writing on prep CMS_CONDITIONS account 
#                this could be dropped introducing a specific entry in the .netrc 
# Requirement 2: an entry "Dropbox" in the .netrc for the authentication

class DB:
    def __init__(self, serviceName, schemaName ):
        self.serviceName = serviceName
        self.schemaName = schemaName
        self.connStr = None

    def connect( self ):
        command = "cmscond_authentication_manager -s %s --list_conn | grep '%s@%s'" %(self.serviceName,self.schemaName,self.serviceName)
        pipe = subprocess.Popen( command, shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        out = pipe.communicate()[0]
        srvconn = '%s@%s' %(self.schemaName,self.serviceName)
        rowpwd = out.split(srvconn)[1].split(self.schemaName)[1]
        pwd = ''
        for c in rowpwd:
            if c != ' ' and c != '\n':
                pwd += c
        self.connStr =  '%s/%s@%s' %(self.schemaName,pwd,self.serviceName)

    def setSynchronizationType( self, tag, synchType ):
        db = cx_Oracle.connect(self.connStr)
        cursor = db.cursor()
        db.begin()
        cursor.execute('UPDATE TAG SET SYNCHRONIZATION =:SYNCH WHERE NAME =:NAME',(synchType,tag,))
        db.commit()

    def getLastInsertedSince( self, tag, snapshot ):
        db = cx_Oracle.connect(self.connStr)
        cursor = db.cursor()
        cursor.execute('SELECT SINCE, INSERTION_TIME FROM IOV WHERE TAG_NAME =:TAG_NAME AND INSERTION_TIME >:TIME ORDER BY INSERTION_TIME DESC',(tag,snapshot))
        row = cursor.fetchone()
        return row

    def removeTag( self, tag ):
        db = cx_Oracle.connect(self.connStr)
        cursor = db.cursor()
        db.begin()
        cursor.execute('DELETE FROM IOV WHERE TAG_NAME =:TAG_NAME',(tag,))
        cursor.execute('DELETE FROM TAG WHERE NAME=:NAME',(tag,))
        db.commit()

def makeBaseFile( inputTag, startingSince ):
    cwd = os.getcwd()
    baseFile = '%s_%s.db' %(inputTag,startingSince)
    baseFilePath = os.path.join(cwd,baseFile)
    if os.path.exists( baseFile ):
        os.remove( baseFile )
    command = "conddb_import -c sqlite_file:%s -f oracle://cms_orcon_adg/CMS_CONDITIONS -i %s -t %s -b %s" %(baseFile,inputTag,inputTag,startingSince)
    pipe = subprocess.Popen( command, shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    out = pipe.communicate()[0]
    if not os.path.exists( baseFile ):
        msg = 'ERROR: base file has not been created: %s' %out
        raise Exception( msg )
    return baseFile
        

def makeMetadataFile( inputTag, destTag, since, description ):
    cwd = os.getcwd()
    metadataFile = os.path.join(cwd,'%s.txt') %destTag
    if os.path.exists( metadataFile ):
        os.remove( metadataFile )
    metadata = {}
    metadata[ "destinationDatabase" ] = "oracle://cms_orcoff_prep/CMS_CONDITIONS"
    tagList = {}
    tagList[ destTag ] = { "dependencies": {}, "synchronizeTo": "any" }
    metadata[ "destinationTags" ] = tagList
    metadata[ "inputTag" ] = inputTag
    metadata[ "since" ] = since
    metadata[ "userText" ] = description
    fileName = destTag+".txt"
    with open( fileName, "w" ) as file:
        file.write(json.dumps(metadata,file,indent=4,sort_keys=True))

def uploadFile( fileName, logFileName ):
    command = "uploadConditions.py %s" %fileName
    pipe = subprocess.Popen( command, shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    out = pipe.communicate()[0]
    lines = out.split('\n')
    ret = False
    for line in lines:
        if line.startswith('\t '):
            if line.startswith('\t status : -2'):
                print 'ERROR: upload of file %s failed.' %fileName
            if line.startswith('\t %s' %fileName):
                returnCode = line.split('\t %s :' %fileName)[1].strip()
                if returnCode == 'True':
                    ret = True
    with open(logFileName,'a') as logFile:
        logFile.write(out)
    return ret

class UploadTest:
    def __init__(self, db):
        self.db = db
        self.errors = 0
        self.logFileName = 'conditionUloadTest.log'

    def log( self, msg ):
        print msg
        with open(self.logFileName,'a') as logFile:
            logFile.write(msg)
            logFile.write('\n')

    def upload( self, inputTag, baseFile, destTag, synchro, destSince, success, expectedAction ):
        insertedSince = None
        destFile = '%s.db' %destTag
        metaDestFile = '%s.txt' %destTag
        shutil.copyfile( baseFile, destFile )
        self.log( '# ---------------------------------------------------------------------------')
        self.log( '# Testing tag %s with synch=%s, destSince=%s - expecting ret=%s action=%s' %(destTag,synchro,destSince,success,expectedAction))
    
        descr = 'Testing conditionsUpload with synch:%s - expected action: %s' %(synchro,expectedAction)
        makeMetadataFile( inputTag, destTag, destSince, descr )
        beforeUpload = datetime.datetime.utcnow()
        ret = uploadFile( destFile, self.logFileName )
        if ret != success:
            self.log( 'ERROR: the return value for the upload of tag %s with sychro %s was %s, while the expected result is %s' %(destTag,synchro,ret,success))
            self.errors += 1
        else:
            row = self.db.getLastInsertedSince( destTag, beforeUpload )
            if ret == True:
                if expectedAction == 'CREATE' or expectedAction == 'INSERT' or expectedAction == 'APPEND':
                    if destSince != row[0]:
                        self.log( 'ERROR: the since inserted is %s, expected value is %s - expected action: %s' %(row[0],destSince,expectedAction))
                        self.errors += 1
                    else:
                        self.log( '# OK: Found expected value for last since inserted: %s timestamp: %s' %(row[0],row[1]))
                        insertedSince = row[0]
                elif expectedAction == 'SYNCHRONIZE':
                    if destSince == row[0]:
                        self.log( 'ERROR: the since inserted %s has not been synchronized with the FCSR - expected action: %s' %(row[0],expectedAction))
                        self.errors += 1
                    else:
                        self.log( '# OK: Found synchronized value for the last since inserted: %s timestamp: %s' %(row[0],row[1]))
                        insertedSince = row[0]
                else:
                    self.log( 'ERROR: found an appended since %s - expected action: %s' %(row[0],expectedAction))
                    self.errors += 1
            else:
                if not row is None:
                    self.log( 'ERROR: found new insered since: %s timestamp: %s' %(row[0],row[1]))
                    self.errors += 1
                if expectedAction != 'FAIL':
                    self.log( 'ERROR: Upload failed. Expected value: %s' %(destSince))
                    self.errors += 1
                else:
                    self.log( '# OK: Upload failed as expected.')
        os.remove( destFile )
        os.remove( metaDestFile )
        return insertedSince


def main():
    print 'Testing...'
    serviceName = 'cms_orcoff_prep'
    schemaName = 'CMS_CONDITIONS'
    db = DB(serviceName,schemaName)
    db.connect()
    inputTag = 'runinfo_31X_mc'
    bfile0 = makeBaseFile( inputTag,1)
    bfile1 = makeBaseFile( inputTag,100)
    test = UploadTest( db )
    # test with synch=any
    tag = 'test_CondUpload_any'
    test.upload( inputTag, bfile0, tag, 'any', 1, True, 'CREATE' )  
    test.upload( inputTag, bfile1, tag, 'any', 1, False, 'FAIL' )  
    test.upload( inputTag, bfile0, tag, 'any', 200, True, 'APPEND' )  
    test.upload( inputTag, bfile0, tag, 'any', 100, True, 'INSERT')  
    test.upload( inputTag, bfile0, tag, 'any', 200, True, 'INSERT')  
    db.removeTag( tag )
    # test with synch=validation
    tag = 'test_CondUpload_validation'
    test.upload( inputTag, bfile0, tag, 'validation', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'validation' ) 
    test.upload( inputTag, bfile0, tag, 'validation', 1, True, 'INSERT')  
    test.upload( inputTag, bfile0, tag, 'validation', 200, True, 'APPEND')  
    test.upload( inputTag, bfile0, tag, 'validation', 100, True, 'INSERT')  
    db.removeTag( tag )
    # test with synch=mc
    tag = 'test_CondUpload_mc'
    test.upload( inputTag, bfile1, tag, 'mc', 1, False, 'FAIL')  
    test.upload( inputTag, bfile0, tag, 'mc', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'mc' ) 
    test.upload( inputTag, bfile0, tag, 'mc', 1, False, 'FAIL')  
    test.upload( inputTag, bfile0, tag, 'mc', 200, False, 'FAIL') 
    db.removeTag( tag )
    # test with synch=hlt
    tag = 'test_CondUpload_hlt'
    test.upload( inputTag, bfile0, tag, 'hlt', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'hlt' ) 
    test.upload( inputTag, bfile0, tag, 'hlt', 200, True, 'SYNCHRONIZE')  
    fcsr = test.upload( inputTag, bfile0, tag, 'hlt', 100, True, 'SYNCHRONIZE')  
    if not fcsr is None:
        since = fcsr + 200
        test.upload( inputTag, bfile0, tag, 'hlt', since, True, 'APPEND')  
        since = fcsr + 100
        test.upload( inputTag, bfile0, tag, 'hlt', since, True, 'INSERT')  
    db.removeTag( tag )
    # test with synch=express
    tag = 'test_CondUpload_express'
    test.upload( inputTag, bfile0, tag, 'express', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'express' ) 
    test.upload( inputTag, bfile0, tag, 'express', 200, True, 'SYNCHRONIZE')  
    fcsr = test.upload( inputTag, bfile0, tag, 'express', 100, True, 'SYNCHRONIZE')  
    if not fcsr is None:
        since = fcsr + 200
        test.upload( inputTag, bfile0, tag, 'express', since, True, 'APPEND')  
        since = fcsr + 100
        test.upload( inputTag, bfile0, tag, 'express', since, True, 'INSERT')  
    db.removeTag( tag )
    # test with synch=prompt
    tag = 'test_CondUpload_prompt'
    test.upload( inputTag, bfile0, tag, 'prompt', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'prompt' ) 
    test.upload( inputTag, bfile0, tag, 'prompt', 200, True, 'SYNCHRONIZE')  
    fcsr = test.upload( inputTag, bfile0, tag, 'prompt', 100, True, 'SYNCHRONIZE')  
    if not fcsr is None:
        since = fcsr + 200
        test.upload( inputTag, bfile0, tag, 'prompt', since, True, 'APPEND')  
        since = fcsr + 100
        test.upload( inputTag, bfile0, tag, 'prompt', since, True, 'INSERT')  
    db.removeTag( tag )
    # test with synch=pcl
    tag = 'test_CondUpload_pcl'
    test.upload( inputTag, bfile0, tag, 'pcl', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'pcl' ) 
    test.upload( inputTag, bfile0, tag, 'pcl', 200, False, 'FAIL')  
    if not fcsr is None:
        since = fcsr + 200
        test.upload( inputTag, bfile0, tag, 'pcl', since, True, 'APPEND')  
        since = fcsr + 100
        test.upload( inputTag, bfile0, tag, 'pcl', since, True, 'INSERT')  
    db.removeTag( tag )
    # test with synch=offline
    tag = 'test_CondUpload_offline'
    test.upload( inputTag, bfile0, tag, 'offline', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'offline' ) 
    test.upload( inputTag, bfile0, tag, 'offline', 1000, True, 'APPEND')
    test.upload( inputTag, bfile0, tag, 'offline', 500, False, 'FAIL' ) 
    test.upload( inputTag, bfile0, tag, 'offline', 1000, False, 'FAIL' ) 
    test.upload( inputTag, bfile0, tag, 'offline', 2000, True, 'APPEND' ) 
    db.removeTag( tag )
    # test with synch=runmc
    tag = 'test_CondUpload_runmc'
    test.upload( inputTag, bfile0, tag, 'runmc', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'runmc' ) 
    test.upload( inputTag, bfile0, tag, 'runmc', 1000, True, 'APPEND')
    test.upload( inputTag, bfile0, tag, 'runmc', 500, False, 'FAIL' ) 
    test.upload( inputTag, bfile0, tag, 'runmc', 1000, False, 'FAIL' ) 
    test.upload( inputTag, bfile0, tag, 'runmc', 2000, True, 'APPEND' ) 
    db.removeTag( tag )
    os.remove( bfile0 )
    os.remove( bfile1 )
    print 'Done. Errors: %s' %test.errors
    
if __name__ == '__main__':
    main()
