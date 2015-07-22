import subprocess
import json
import netrc
from os import rename
from os import remove
from os import path

confFileName ='popcon2dropbox.json'
fileNameForDropBox = 'input_for_dropbox'
dbFileForDropBox = '%s.db' %fileNameForDropBox
dbLogFile = '%s_log.db' %fileNameForDropBox

import upload_popcon

class CondMetaData(object):
   def __init__( self ):
      self.md = {}
      with open(confFileName) as jf:
         self.md = json.load(jf)
         
   def authPath( self ):
      return self.md.get('authenticationPath')

   def authSys( self ):
      return self.md.get('authenticationSystem')

   def destinationDatabase( self ):
      return self.md.get('destinationDatabase')

   def logDbFileName( self ):
      return self.md.get('logDbFileName')

   def synchronizeTo( self ):
      return self.md.get('synchronizeTo')

   def records( self ):
      return self.md.get('records')

   def dumpMetadataForUpload( self, inputtag, desttag, comment ):
      uploadMd = {}
      uploadMd['destinationDatabase'] = self.destinationDatabase()
      tags = {}
      tagInfo = {}
      tagInfo['dependencies'] = {}
      tagInfo['synchronizeTo'] = self.synchronizeTo()
      tags[ desttag ] = tagInfo
      print tags
      uploadMd['destinationTags'] = tags
      uploadMd['inputTag'] = inputtag
      uploadMd['since'] = None
      uploadMd['userText'] = comment
      with open( '%s.txt' %fileNameForDropBox, 'wb') as jf:
         jf.write( json.dumps( uploadMd, sort_keys=True, indent = 2 ) )
         jf.write('\n')

def runO2O( cmsswdir, releasepath, release, arch, jobfilename, logfilename, *p ):
    # first remove any existing metadata file...
    if path.exists( '%s.db' %fileNameForDropBox ):
       print "Removing files with name %s" %fileNameForDropBox
       remove( '%s.db' %fileNameForDropBox )
    if path.exists( '%s.txt' %fileNameForDropBox ):
       remove( '%s.txt' %fileNameForDropBox )
    command = 'export SCRAM_ARCH=%s;' %arch
    command += 'CMSSWDIR=%s;' %cmsswdir
    command += 'source ${CMSSWDIR}/cmsset_default.sh;'
    command += 'cd %s/%s/src;' %(releasepath,release)
    command += 'eval `scramv1 runtime -sh`;'
    command += 'cd -;'
    command += 'pwd;'
    command += 'cmsRun %s ' %jobfilename
    command += ' '.join(p)
    command += ' 2>&1'
    pipe = subprocess.Popen( command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
    stdout_val = pipe.communicate()[0]
    return stdout_val

def upload_to_dropbox( backend ):
    md = CondMetaData()
    # first remove any existing metadata file...
    if path.exists( '%s.txt' %fileNameForDropBox ):
       remove( '%s.txt' %fileNameForDropBox )
    try:
       dropBox = upload_popcon.DropBox(upload_popcon.defaultHostname, upload_popcon.defaultUrlTemplate)
       # Try to find the netrc entry
       try:
          (username, account, password) = netrc.netrc().authenticators(upload_popcon.defaultNetrcHost)
       except Exception:
          print 'Netrc entry "DropBox" not found.'
          return
       print 'signing in...'
       dropBox.signIn(username, password)
       print 'signed in'
       for k,v in  md.records().items():
          destTag = v.get("destinationTag")
          inputTag = v.get("sqliteTag")
          if inputTag == None:
             inputTag = destTag
          comment = v.get("comment")
          md.dumpMetadataForUpload( inputTag, destTag, comment )
          dropBox.uploadFile(dbFileForDropBox, backend, upload_popcon.defaultTemporaryFile)
       dropBox.signOut()
    except upload_popcon.HTTPError as e:
       print e

