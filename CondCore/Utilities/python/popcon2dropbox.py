import subprocess
import json
import netrc
import sqlite3
import os
import shutil
from datetime import datetime

confFileName ='popcon2dropbox.json'
fileNameForDropBox = 'input_for_dropbox'
dbFileForDropBox = '%s.db' %fileNameForDropBox
dbLogFile = '%s_log.db' %fileNameForDropBox
errorInUploadFileFolder = 'upload_errors'
dateformatForFolder = "%y-%m-%d-%H-%M-%S"
dateformatForLabel = "%y-%m-%d %H:%M:%S"

"""
import upload_popcon

class CondMetaData(object):
   def __init__( self, fileName ):
      self.md = {}
      self.datef = datetime.now()
      with open(fileName) as jf:
         try:
            self.md = json.load(jf)
         except ValueError as e:
            errorMessage = 'CondMetaData.__init__: Problem in decoding JSON file. Original error: ' + e.message
            raise ValueError(errorMessage)

   def authPath( self ):
      apath = ''
      if self.md.has_key('authenticationPath'):
         apath = self.md.get('authenticationPath')
      return apath

   def authSys( self ):
      asys = 1
      if self.md.has_key('authenticationSys'):
         asys = self.md.get('authenticationSystem')
      return asys

   def destinationDatabase( self ):
      return self.md.get('destinationDatabase')

   def logDbFileName( self ):
      return self.md.get('logDbFileName')

   def records( self ):
      return self.md.get('records')

   def dumpMetadataForUpload( self, inputtag, desttag, comment ):
      uploadMd = {}
      uploadMd['destinationDatabase'] = self.destinationDatabase()
      tags = {}
      tagInfo = {}
      tags[ desttag ] = tagInfo
      uploadMd['destinationTags'] = tags
      uploadMd['inputTag'] = inputtag
      uploadMd['since'] = None
      datelabel = self.datef.strftime(dateformatForLabel)
      uploadMd['userText'] = '%s : %s' %(datelabel,comment)
      with open( '%s.txt' %fileNameForDropBox, 'wb') as jf:
         jf.write( json.dumps( uploadMd, sort_keys=True, indent = 2 ) )
         jf.write('\n')
def dumpMetadataForUpload( destDb, destTag, comment ):
    uploadMd = {}
    uploadMd['destinationDatabase'] = destDb
    tags = {}
    tagInfo = {}
    tags[ destTag ] = tagInfo
    uploadMd['destinationTags'] = tags
    uploadMd['inputTag'] = destTag
    uploadMd['since'] = None
    datef = datetime.now() 
    #datelabel = datef.strftime(dateformatForLabel)
    uploadMd['userText'] = '%s : %s' %(datelabel,comment)
    with open( '%s.txt' %fileNameForDropBox, 'wb') as jf:
       jf.write( json.dumps( uploadMd, sort_keys=True, indent = 2 ) )
       jf.write('\n')
"""
   
def upload( destDb, destTag, comment, authPath ):
    #md = CondMetaData(cfileName)
    datef = datetime.now()

    # check if the expected input file is there...                                                                                                                              
    if not os.path.exists( dbFileForDropBox ):
       print 'The input sqlite file has not been produced.'
       return -1

    empty = True
    try:
       dbcon = sqlite3.connect( dbFileForDropBox )
       dbcur = dbcon.cursor()
       dbcur.execute('SELECT * FROM IOV')
       rows = dbcur.fetchall()
       for r in rows:
          empty = False
       dbcon.close()
       if empty:
          print 'The input sqlite file produced contains no data. The upload will be skipped.'
          return 0
    except Exception as e:
       print 'Check on input data failed: %s' %str(e)
       return -2

    # first remove any existing metadata file...                                                                                                                                
    if os.path.exists( '%s.txt' %fileNameForDropBox ):
       os.remove( '%s.txt' %fileNameForDropBox )
    
    ret = 0
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
    with open( '%s.txt' %fileNameForDropBox, 'wb') as jf:
       jf.write( json.dumps( uploadMd, sort_keys=True, indent = 2 ) )
       jf.write('\n')

    # run the upload
    uploadCommand = 'uploadConditions.py %s' %fileNameForDropBox
    if not authPath is None:
       uploadCommand += ' -a %s' %authPath
    try:
       pipe = subprocess.Popen( uploadCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
       stdout = pipe.communicate()[0]
       print stdout
       retCode = pipe.returncode
       if retCode != 0:
          # save a copy of the files in case of upload failure...
          leafFolderName = datef.strftime(dateformatForFolder)
          fileFolder = os.path.join( errorInUploadFileFolder, leafFolderName)
          if not os.path.exists(fileFolder):
             os.makedirs(fileFolder)
          df= '%s.db' %fileNameForDropBox
          mf= '%s.txt' %fileNameForDropBox
          dataDestFile = os.path.join( fileFolder, df)
          if not os.path.exists(dataDestFile):
             shutil.copy2(df, dataDestFile)
          shutil.copy2(mf,os.path.join(fileFolder,mf))
          print "Upload failed. Data file and metadata saved in folder '%s'" %os.path.abspath(fileFolder)
       ret |= retCode
    except Exception as e:
       ret |= 1
       print e
    return ret

def run( args ):
    if os.path.exists( '%s.db' %fileNameForDropBox ):
       print "Removing files with name %s" %fileNameForDropBox
       os.remove( '%s.db' %fileNameForDropBox )
    if os.path.exists( '%s.txt' %fileNameForDropBox ):
       os.remove( '%s.txt' %fileNameForDropBox )
    command = 'cmsRun %s ' %args.job_file
    command += ' destinationDatabase=%s' %args.destDb
    command += ' destinationTag=%s' %args.destTag
    command += ' 2>&1'
    pipe = subprocess.Popen( command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
    stdout = pipe.communicate()[0]
    retCode = pipe.returncode
    print stdout
    print 'Return code is: %s' %retCode
    if retCode!=0:
       print 'O2O job failed. Skipping upload.'
       return retCode
    return upload( args.destDb, args.destTag, args.comment, args.auth )
