import subprocess
import json
from os import rename
from os import remove
from os import path

confFileName ='popcon2dropbox.json'
fileNameForDropBox = 'input_for_dropbox'
dbFileForDropBox = '%s.db' %fileNameForDropBox
dbLogFile = '%s_log.db' %fileNameForDropBox

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

   def records( self ):
      return self.md.get('records')

   def dumpMetadataForUpload( self, inputtag, desttag, comment ):
      uploadMd = {}
      uploadMd['destinationDatabase'] = self.destinationDatabase()
      tags = {}
      tagInfo = {}
      tagInfo['dependencies'] = {}
      tagInfo['synchronizeTo'] = 'offline'
      tags[ desttag ] = tagInfo
      print tags
      uploadMd['destinationTags'] = tags
      uploadMd['inputTag'] = inputtag
      uploadMd['since'] = None
      uploadMd['userText'] = comment
      with open( '%s.txt' %fileNameForDropBox, 'wb') as jf:
         jf.write( json.dumps( uploadMd, sort_keys=True, indent = 2 ) )
         jf.write('\n')

def runO2O( cmsswdir, releasepath, release, arch, jobfilename, logfilename ):
    command = 'export SCRAM_ARCH=%s;' %arch
    command += 'CMSSWDIR=%s;' %cmsswdir
    command += 'source ${CMSSWDIR}/cmsset_default.sh;'
    command += 'cd %s/%s/src;' %(releasepath,release)
    command += 'eval `scramv1 runtime -sh`;'
    command += 'cd -;'
    command += 'pwd;'
    command += 'cmsRun %s 2>&1' %jobfilename
    print '## about to execute: %s' %command
    pipe = subprocess.Popen( command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
    stdout_val = pipe.communicate()[0]
    return stdout_val

def upload_to_dropbox():
    md = CondMetaData()
    # first remove any existing file...
    if path.exists( '%s.txt' %fileNameForDropBox ):
       remove( '%s.txt' %fileNameForDropBox )
    for k,v in  md.records().items():
       destTag = v.get("destinationTag")
       inputTag = v.get("sqliteTag")
       if inputTag == None:
          inputTag = destTag
       comment = v.get("comment")
       md.dumpMetadataForUpload( inputTag, destTag, comment )
       command = 'export http_proxy=http://cmsproxy.cms:3128/;'
       command += 'export https_proxy=https://cmsproxy.cms:3128/;'
       command += 'python upload_popcon.py %s -b offline' %dbFileForDropBox
       print 'executing command:%s' %command
       pipe = subprocess.Popen(  command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
       print pipe.communicate()[0]
       rename( '%s.txt' %fileNameForDropBox, '%s.txt' %destTag )
       




