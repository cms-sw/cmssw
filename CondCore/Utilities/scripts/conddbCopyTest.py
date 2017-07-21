#!/usr/bin/env python

import sqlite3
import subprocess
import json
import os
import shutil
import datetime

fileName = 'copy_test.db'

class DB:
    def __init__(self):
        pass

    def setSynchronizationType( self, tag, synchType ):
        db = sqlite3.connect(fileName)
        cursor = db.cursor()
        cursor.execute('UPDATE TAG SET SYNCHRONIZATION =? WHERE NAME =?',(synchType,tag,))
        db.commit()

    def getLastInsertedSince( self, tag, snapshot ):
        db = sqlite3.connect(fileName)
        cursor = db.cursor()
        cursor.execute('SELECT SINCE, INSERTION_TIME FROM IOV WHERE TAG_NAME =? AND INSERTION_TIME >? ORDER BY INSERTION_TIME DESC',(tag,snapshot))
        row = cursor.fetchone()
        return row

def prepareFile( inputTag, sourceTag, startingSince ):
    command = "conddb --yes copy %s %s --destdb %s -f %s" %(inputTag,sourceTag,fileName,startingSince)
    pipe = subprocess.Popen( command, shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    out = pipe.communicate()[0]

def copy( sourceTag, destTag, since, logFileName ):
    command = "conddb --yes --db %s copy %s %s -f %s --synchronize" %(fileName,sourceTag,destTag,since)
    pipe = subprocess.Popen( command, shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    out = pipe.communicate()[0]
    lines = out.split('\n')
    ret = pipe.returncode
    for line in lines:
        print line
    with open(logFileName,'a') as logFile:
        logFile.write(out)
    return ret==0

class CopyTest:
    def __init__(self, db):
        self.db = db
        self.errors = 0
        self.logFileName = 'conddbCopyTest.log'

    def log( self, msg ):
        print msg
        with open(self.logFileName,'a') as logFile:
            logFile.write(msg)
            logFile.write('\n')

    def execute( self, sourceTag, baseFile, destTag, synchro, destSince, success, expectedAction ):
        insertedSince = None
        metaDestFile = '%s.txt' %destTag
        #shutil.copyfile( baseFile, destFile )
        self.log( '# ---------------------------------------------------------------------------')
        self.log( '# Testing tag %s with synch=%s, destSince=%s - expecting ret=%s action=%s' %(destTag,synchro,destSince,success,expectedAction))
    
        descr = 'Testing conditionsUpload with synch:%s - expected action: %s' %(synchro,expectedAction)
        beforeUpload = datetime.datetime.utcnow()
        ret = copy( sourceTag, destTag, destSince, self.logFileName )
        if ret != success:
            self.log( 'ERROR: the return value for the copy of tag %s with sychro %s was %s, while the expected result is %s' %(destTag,synchro,ret,success))
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
        return insertedSince


def main():
    print 'Testing...'
    bfile0 = fileName
    bfile1 = fileName
    db = DB()
    inputTag = 'runinfo_31X_mc'
    inputTag0  ='runinfo_0'
    inputTag1 = 'runinfo_1'
    prepareFile( inputTag,inputTag0,1)
    prepareFile( inputTag,inputTag1,100)
    test = CopyTest( db )
    # test with synch=any
    tag = 'test_CondUpload_any'
    test.execute( inputTag0, bfile0, tag, 'any', 1, True, 'CREATE' )
    test.execute( inputTag0, bfile0, tag, 'any', 200, True, 'APPEND' )  
    test.execute( inputTag0, bfile0, tag, 'any', 100, True, 'INSERT')  
    test.execute( inputTag0, bfile0, tag, 'any', 200, True, 'INSERT')  
    # test with synch=validation
    tag = 'test_CondUpload_validation'
    test.execute( inputTag0, bfile0, tag, 'validation', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'validation' ) 
    test.execute( inputTag0, bfile0, tag, 'validation', 1, True, 'INSERT')  
    test.execute( inputTag0, bfile0, tag, 'validation', 200, True, 'APPEND')  
    test.execute( inputTag0, bfile0, tag, 'validation', 100, True, 'INSERT')  
    # test with synch=mc
    tag = 'test_CondUpload_mc'
    test.execute( inputTag0, bfile0, tag, 'mc', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'mc' ) 
    test.execute( inputTag1, bfile1, tag, 'mc', 1, False, 'FAIL')  
    test.execute( inputTag0, bfile0, tag, 'mc', 1, False, 'FAIL')  
    test.execute( inputTag0, bfile0, tag, 'mc', 200, False, 'FAIL') 
    # test with synch=hlt
    tag = 'test_CondUpload_hlt'
    test.execute( inputTag0, bfile0, tag, 'hlt', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'hlt' ) 
    test.execute( inputTag0, bfile0, tag, 'hlt', 200, True, 'SYNCHRONIZE')  
    fcsr = test.execute( inputTag0, bfile0, tag, 'hlt', 100, True, 'SYNCHRONIZE')  
    if not fcsr is None:
        since = fcsr + 200
        test.execute( inputTag0, bfile0, tag, 'hlt', since, True, 'APPEND')  
        since = fcsr + 100
        test.execute( inputTag0, bfile0, tag, 'hlt', since, True, 'INSERT')  
    # test with synch=express
    tag = 'test_CondUpload_express'
    test.execute( inputTag0, bfile0, tag, 'express', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'express' ) 
    test.execute( inputTag0, bfile0, tag, 'express', 200, True, 'SYNCHRONIZE')  
    fcsr = test.execute( inputTag0, bfile0, tag, 'express', 100, True, 'SYNCHRONIZE')  
    if not fcsr is None:
        since = fcsr + 200
        test.execute( inputTag0, bfile0, tag, 'express', since, True, 'APPEND')  
        since = fcsr + 100
        test.execute( inputTag0, bfile0, tag, 'express', since, True, 'INSERT')  
    # test with synch=prompt
    tag = 'test_CondUpload_prompt'
    test.execute( inputTag0, bfile0, tag, 'prompt', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'prompt' ) 
    test.execute( inputTag0, bfile0, tag, 'prompt', 200, True, 'SYNCHRONIZE')  
    fcsr = test.execute( inputTag0, bfile0, tag, 'prompt', 100, True, 'SYNCHRONIZE')  
    if not fcsr is None:
        since = fcsr + 200
        test.execute( inputTag0, bfile0, tag, 'prompt', since, True, 'APPEND')  
        since = fcsr + 100
        test.execute( inputTag0, bfile0, tag, 'prompt', since, True, 'INSERT')  
    # test with synch=pcl
    tag = 'test_CondUpload_pcl'
    test.execute( inputTag0, bfile0, tag, 'pcl', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'pcl' ) 
    test.execute( inputTag0, bfile0, tag, 'pcl', 200, False, 'FAIL')  
    if not fcsr is None:
        since = fcsr + 200
        test.execute( inputTag0, bfile0, tag, 'pcl', since, True, 'APPEND')  
        since = fcsr + 100
        test.execute( inputTag0, bfile0, tag, 'pcl', since, True, 'INSERT')  
    # test with synch=offline
    tag = 'test_CondUpload_offline'
    test.execute( inputTag0, bfile0, tag, 'offline', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'offline' ) 
    test.execute( inputTag0, bfile0, tag, 'offline', 1000, True, 'APPEND')
    test.execute( inputTag0, bfile0, tag, 'offline', 500, False, 'FAIL' ) 
    test.execute( inputTag0, bfile0, tag, 'offline', 1000, False, 'FAIL' ) 
    test.execute( inputTag0, bfile0, tag, 'offline', 2000, True, 'APPEND' ) 
    # test with synch=runmc
    tag = 'test_CondUpload_runmc'
    test.execute( inputTag0, bfile0, tag, 'runmc', 1, True, 'CREATE')  
    db.setSynchronizationType( tag, 'runmc' ) 
    test.execute( inputTag0, bfile0, tag, 'runmc', 1000, True, 'APPEND')
    test.execute( inputTag0, bfile0, tag, 'runmc', 500, False, 'FAIL' ) 
    test.execute( inputTag0, bfile0, tag, 'runmc', 1000, False, 'FAIL' ) 
    test.execute( inputTag0, bfile0, tag, 'runmc', 2000, True, 'APPEND' )
    os.remove( fileName )
    print 'Done. Errors: %s' %test.errors

    
if __name__ == '__main__':
    main()
