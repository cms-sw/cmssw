#! /usr/bin/env python

from __future__ import print_function
print('Starting cmsLHEtoEOSManager.py')

__version__ = "$Revision: 1.13 $"

import os
import subprocess
import time
import re

defaultEOSRootPath = '/eos/cms/store/lhe'
if "CMSEOS_LHE_ROOT_DIRECTORY" in os.environ:
  defaultEOSRootPath = os.environ["CMSEOS_LHE_ROOT_DIRECTORY"]
defaultEOSLoadPath = 'root://eoscms.cern.ch/'
defaultEOSlistCommand = 'xrdfs '+defaultEOSLoadPath+' ls '
defaultEOSmkdirCommand = 'xrdfs '+defaultEOSLoadPath+' mkdir '
defaultEOSfeCommand = 'xrdfs '+defaultEOSLoadPath+' stat -q IsReadable '
defaultEOSchecksumCommand = 'xrdfs '+defaultEOSLoadPath+' query checksum '
defaultEOScpCommand = 'xrdcp -np '

def findXrdDir(theDirRecord):

    elements = theDirRecord.split(' ')
    if len(elements):
        return elements[-1].rstrip('\n').split('/')[-1]
    else:
        return None

def articleExist(artId):

    itExists = False
    theCommand = defaultEOSlistCommand+' '+defaultEOSRootPath
    dirList = subprocess.Popen(["/bin/sh","-c",theCommand], stdout=subprocess.PIPE, universal_newlines=True)
    for line in dirList.stdout.readlines():
        if findXrdDir(line) == str(artId): 
            itExists = True

    return itExists

def lastArticle():

    artList = [0]

    theCommand = defaultEOSlistCommand+' '+defaultEOSRootPath
    dirList = subprocess.Popen(["/bin/sh","-c",theCommand], stdout=subprocess.PIPE, universal_newlines=True)
    for line in dirList.stdout.readlines():
        try:
            if line.rstrip('\n') != '':
                artList.append(int(findXrdDir(line)))
        except:
            break

    return max(artList)


def fileUpload(uploadPath,lheList, checkSumList, reallyDoIt, force=False):

    inUploadScript = ''
    index = 0
    for f in lheList:
        realFileName = f.split('/')[-1]
        # Check the file existence
        newFileName = uploadPath+'/'+str(realFileName)
        addFile = True
        additionalOption = ''  
        theCommand = defaultEOSfeCommand+' '+newFileName
        exeFullList = subprocess.Popen(["/bin/sh","-c",theCommand], stdout=subprocess.PIPE, universal_newlines=True)
        result = exeFullList.stdout.readlines()
        if [line for line in result if ("flags:" in line.lower()) and ("isreadable" in line.lower())] and (not force):
            addFile = False
            print('File '+newFileName+' already exists: do you want to overwrite? [y/n]')
            reply = raw_input()
            if reply == 'y' or reply == 'Y':
                addFile = True
                additionalOption = ' -f '
                print('')
                print('Overwriting file '+newFileName+'\n')
        # add the file
        if addFile:
#            print 'Adding file '+str(f)+'\n'
            inUploadScript = defaultEOScpCommand + additionalOption + ' ' + str(f) + ' ' + defaultEOSLoadPath+uploadPath + '/' + str(realFileName)
            print('Uploading file %s...' % str(f))
            if reallyDoIt:
                exeRealUpload = subprocess.Popen(["/bin/sh","-c",inUploadScript])
                exeRealUpload.communicate()
                eosCheckSumCommand = defaultEOSchecksumCommand + uploadPath + '/' + str(realFileName) + ' | awk \'{print $2}\' | cut -d= -f2'
                exeEosCheckSum = subprocess.Popen(eosCheckSumCommand ,shell=True, stdout=subprocess.PIPE, universal_newlines=True)
                EosCheckSum = exeEosCheckSum.stdout.read()
                assert exeEosCheckSum.wait() == 0
               # print 'checksum: eos = ' + EosCheckSum + 'orig file = ' + checkSumList[index] + '\n'
                if checkSumList[index] not in EosCheckSum:
                    print('WARNING! The checksum for file ' + str(realFileName) + ' in EOS\n')
                    print(EosCheckSum + '\n')
                    print('does not match the checksum of the original one\n')
                    print(checkSumList[index] + '\n')
                    print('please try to re-upload file ' + str(realFileName) + ' to EOS.\n')
                else:
                    print('Checksum OK for file ' + str(realFileName))
        index = index+1
 
# launch the upload shell script        

#    print '\n Launching upload script \n'+inUploadScript+'\n at '+time.asctime(time.localtime(time.time()))+' ...\n'
#    if reallyDoIt:  
#      exeRealUpload = subprocess.Popen(["/bin/sh","-c",inUploadScript])
#      exeRealUpload.communicate()
    print('\n Upload ended at '+time.asctime(time.localtime(time.time())))

#################################################################################################    
        
if __name__ == '__main__':
    
    import optparse
    
    # Here we define an option parser to handle commandline options..
    usage='cmsLHEtoEOSManager.py <options>'
    parser = optparse.OptionParser(usage)
    parser.add_option('-f', '--file',
                      help='LHE local file list to be uploaded, separated by ","' ,
                      default='',
                      dest='fileList')

    parser.add_option('-F', '--files-from', metavar = 'FILE',
                      help='File containing the list of LHE local files be uploaded, one file per line')

    parser.add_option('-n', '--new', 
                      help='Create a new article' ,
                      action='store_true',
                      default=False,
                      dest='newId')                      

    parser.add_option('-u', '--update', 
                      help='Update the article <Id>' ,
                      default=0,
                      type=int,
                      dest='artIdUp')                      

    parser.add_option('-l', '--list', 
                      help='List the files in article <Id>' ,
                      default=0,
                      type=int,
                      dest='artIdLi')                     
    
    parser.add_option('-d', '--dry-run',
                      help='dry run, it does nothing, but you can see what it would do',
                      action='store_true',
                      default=False,
                      dest='dryRun')
    
    parser.add_option('-c', '--compress',
                      help='compress the local .lhe file with xz before upload',
                      action='store_true',
                      default=False,
                      dest='compress')

    parser.add_option('--force',
                      help='Force update if file already exists.',
                      action='store_true',
                      default=False,
                      dest='force')

    (options,args) = parser.parse_args()

    # print banner

    print('')
    print('cmsLHEtoEOSmanager '+__version__[1:-1])
    print('')
    print('Running on ',time.asctime(time.localtime(time.time())))
    print('')
    
    reallyDoIt = not options.dryRun

    # Now some fault control. If an error is found we raise an exception
    if not options.newId and options.artIdUp==0 and options.artIdLi==0:
        raise Exception('Please specify the action to be taken, either "-n", "-u" or "-l"!')
    
    if options.fileList == '' and not options.files_from and (options.newId or options.artIdUp!=0):
        raise Exception('Please provide the input file list!')

    if (options.newId and (options.artIdUp != 0 or options.artIdLi != 0)) or (options.artIdUp != 0 and options.artIdLi != 0):
        raise Exception('Options "-n", "-u" and "-l" are mutually exclusive, please choose only one!')

    if options.newId:
        print('Action: create new article\n')
    elif options.artIdUp != 0:
        print('Action: update article '+str(options.artIdUp)+'\n')
    elif options.artIdLi != 0:
        print('Action: list content of article '+str(options.artIdLi)+'\n')

    if options.artIdLi==0:
        theList = []
        if len(options.fileList) > 0:
            theList=(options.fileList.split(','))
            
        if options.files_from:
            try:
                f = open(options.files_from)
            except IOError:
                raise Exception('Cannot open the file list, \'%s\'' % options.files_from)
            for l in f:
                l = l.strip()
                if len(l) == 0 or l[0] == '#':
                    continue
                theList.append(l)

        theCompressedFilesList = []
        theCheckSumList = []
        for f in theList: 
            # Check the file name extension
            print(f)
            if not ( f.lower().endswith(".lhe") or f.lower().endswith(".lhe.xz") ):
                raise Exception('Input file name must have the "lhe" or "lhe.xz" final extension!')
            if( f.lower().endswith(".lhe.xz") ):
                print("Important! Input file "+f+" is already zipped: please make sure you verified its integrity with xmllint before zipping it. You can do it with:\n")
                print("xmllint file.lhe\n")
                print("Otherwise it is best to pass the unzipped file to this script and let it check its integrity and compress the file with the --compress option\n")
            # Check the local file existence
            if not os.path.exists(f):
                raise Exception('Input file '+f+' does not exists')
            if( f.lower().endswith(".lhe") ):
                theCheckIntegrityCommand = 'xmllint -noout '+f
                exeCheckIntegrity = subprocess.Popen(["/bin/sh","-c", theCheckIntegrityCommand])
                intCode = exeCheckIntegrity.wait()
                if(intCode != 0):
                    raise Exception('Input file '+f+ ' is corrupted')
            if reallyDoIt and options.compress:
              print("Compressing file",f)
              if( f.lower().endswith(".lhe.xz") ):
                  raise Exception('Input file '+f+' is already compressed! This is inconsistent with the --compress option!')
              theCompressionCommand = 'xz '+f
              exeCompression = subprocess.Popen(["/bin/sh","-c",theCompressionCommand])
              exeCompression.communicate()
              theCompressedFilesList.append(f+'.xz')
        if reallyDoIt and options.compress:
          theList = theCompressedFilesList
        for f in theList:
            try:
                exeCheckSum = subprocess.Popen(["/afs/cern.ch/cms/caf/bin/cms_adler32",f], stdout=subprocess.PIPE, universal_newlines=True)
                getCheckSum = subprocess.Popen(["awk", "{print $1}"], stdin=exeCheckSum.stdout, stdout=subprocess.PIPE, universal_newlines=True)
                exeCheckSum.stdout.close()
                output,err = getCheckSum.communicate()
                theCheckSumList.append(output.strip())
            except:
                theCheckSumList.append("missing-adler32")

    newArt = 0
    uploadPath = ''

# new article

    if options.newId:
        oldArt = lastArticle()
        newArt = oldArt+1
        print('Creating new article with identifier '+str(newArt)+' ...\n')
        uploadPath = defaultEOSRootPath+'/'+str(newArt)
        theCommand = defaultEOSmkdirCommand+' '+uploadPath
        if reallyDoIt:
          exeUpload = subprocess.Popen(["/bin/sh","-c",theCommand])
          exeUpload.communicate()

# update article
        
    elif options.artIdUp != 0:
        newArt = options.artIdUp
        if articleExist(newArt):
            uploadPath = defaultEOSRootPath+'/'+str(newArt)
        else:
            raise Exception('Article '+str(newArt)+' to be updated does not exist!')

# list article
        
    elif options.artIdLi !=0:
        listPath = defaultEOSRootPath+'/'+str(options.artIdLi)
        theCommand = defaultEOSlistCommand+' '+listPath
        exeList = subprocess.Popen(["/bin/sh","-c",theCommand], stdout=subprocess.PIPE, universal_newlines=True)
        for line in exeList.stdout.readlines():
            if findXrdDir(line) != None:
                print(findXrdDir(line))


    if newArt > 0:
        fileUpload(uploadPath,theList, theCheckSumList, reallyDoIt, options.force)
        listPath = defaultEOSRootPath+'/'+str(newArt)
        print('')
        print('Listing the '+str(newArt)+' article content after upload:')
        theCommand = defaultEOSlistCommand+' '+listPath
        if reallyDoIt:
          exeFullList = subprocess.Popen(["/bin/sh","-c",theCommand])
          exeFullList.communicate()
        else:
          print('Dry run, nothing was done')
        
