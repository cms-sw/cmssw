#! /usr/bin/env python

__version__ = "$Revision: 1.12 $"

import os
import subprocess
import time
import re

defaultEOSRootPath = '/eos/cms/store/lhe'
defaultEOSLoadPath = 'root://eoscms/'
defaultEOSlistCommand = 'xrd eoscms dirlist '
defaultEOSmkdirCommand = 'xrd eoscms mkdir '
defaultEOSfeCommand = 'xrd eoscms existfile '
defaultEOScpCommand = 'xrdcp -np '

def findXrdDir(theDirRecord):

    elements = theDirRecord.split(' ')
    if len(elements) > 1:
        return elements[-1].rstrip('\n').split('/')[-1]
    else:
        return None

def articleExist(artId):

    itExists = False
    theCommand = defaultEOSlistCommand+' '+defaultEOSRootPath
    dirList = subprocess.Popen(["/bin/sh","-c",theCommand], stdout=subprocess.PIPE)
    for line in dirList.stdout.readlines():
        if findXrdDir(line) == str(artId): 
            itExists = True

    return itExists

def lastArticle():

    artList = [0]

    theCommand = defaultEOSlistCommand+' '+defaultEOSRootPath
    dirList = subprocess.Popen(["/bin/sh","-c",theCommand], stdout=subprocess.PIPE)
    for line in dirList.stdout.readlines():
        try:
            if line.rstrip('\n') != '':
                artList.append(int(findXrdDir(line)))
        except:
            break

    return max(artList)


def fileUpload(uploadPath,lheList, reallyDoIt):

    inUploadScript = ''

    for f in lheList:
        realFileName = f.split('/')[-1]
        # Check the file existence
        newFileName = uploadPath+'/'+str(realFileName)
        addFile = True
        additionalOption = ''  
        theCommand = defaultEOSfeCommand+' '+newFileName
        exeFullList = subprocess.Popen(["/bin/sh","-c",theCommand], stdout=subprocess.PIPE)
        result = exeFullList.stdout.readlines()
        if result[0].rstrip('\n') == 'The file exists.':
            addFile = False
            print 'File '+newFileName+' already exists: do you want to overwrite? [y/n]'
            reply = raw_input()
            if reply == 'y' or reply == 'Y':
                addFile = True
                additionalOption = ' -f '
                print ''
                print 'Overwriting file '+newFileName+'\n'
        # add the file
        if addFile:
            print 'Adding file '+str(f)+'\n'
            inUploadScript += defaultEOScpCommand+additionalOption+' '+str(f)+' '+defaultEOSLoadPath+uploadPath+'/'+str(realFileName)+'\n'

# launch the upload shell script        

    print '\n Launching upload script \n'+inUploadScript+'\n at '+time.asctime(time.localtime(time.time()))+' ...\n'
    if reallyDoIt:  
      exeRealUpload = subprocess.Popen(["/bin/sh","-c",inUploadScript])
      exeRealUpload.communicate()
    print '\n Upload ended at '+time.asctime(time.localtime(time.time()))

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

    parser.add_option('-n', '--new', 
                      help='Create a new article' ,
                      action='store_true',
                      default=False,
                      dest='newId')                      

    parser.add_option('-u', '--update', 
                      help='Update the article <Id>' ,
                      default=0,
                      dest='artIdUp')                      

    parser.add_option('-l', '--list', 
                      help='List the files in article <Id>' ,
                      default=0,
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

    (options,args) = parser.parse_args()

    # print banner

    print ''
    print 'cmsLHEtoEOSmanager '+__version__[1:-1]
    print ''
    print 'Running on ',time.asctime(time.localtime(time.time()))
    print ''
    
    reallyDoIt = not options.dryRun

    # Now some fault control..If an error is found we raise an exception
    if not options.newId and options.artIdUp==0 and options.artIdLi==0:
        raise Exception('Please specify the action to be taken, either "-n", "-u" or "-l"!')
    
    if options.fileList=='' and (options.newId or options.artIdUp!=0):
        raise Exception('Please provide the input file list!')

    if (options.newId and (options.artIdUp != 0 or options.artIdLi != 0)) or (options.artIdUp != 0 and options.artIdLi != 0):
        raise Exception('Options "-n", "-u" and "-l" are mutually exclusive, please chose only one!')

    if options.newId:
        print 'Action: create new article\n'
    elif options.artIdUp != 0:
        print 'Action: update article '+str(options.artIdUp)+'\n'
    elif options.artIdLi != 0:
        print 'Action: list content of article '+str(options.artIdLi)+'\n'

    if options.artIdLi==0:
        theList = options.fileList.split(',')
        theCompressedFilesList = []
        for f in theList: 
            # Check the file name extension
            if not ( f.lower().endswith(".lhe") or f.lower().endswith(".lhe.xz") ):
                raise Exception('Input file name must have the "lhe" or "lhe.xz" final extension!')
            if( f.lower().endswith(".lhe.xz") ):
                print "Important! Input file "+f+" is already zipped: please make sure you verified its integrity with xmllint before zipping it. You can do it with:\n"
                print "xmllint file.lhe\n"
                print "Otherwise it is best to pass the unzipped file to this script and let it check its integrity and compress the file with the --compress option\n"
            # Check the local file existence
            if not os.path.exists(f):
                raise Exception('Input file '+f+' does not exists')
            if( f.lower().endswith(".lhe") ):
                theCheckIntegrityCommand = 'xmllint -noout '+f
                exeCheckIntegrity = subprocess.Popen(["/bin/sh","-c", theCheckIntegrityCommand])
                intCode = exeCheckIntegrity.wait()
                if(intCode is not 0):
                    raise Exception('Input file '+f+ ' is corrupted')
            if reallyDoIt and options.compress:
              print "Compressing file",f
              theCompressionCommand = 'xz '+f
              exeCompression = subprocess.Popen(["/bin/sh","-c",theCompressionCommand])
              exeCompression.communicate()
              theCompressedFilesList.append(f+'.xz')
        if reallyDoIt and options.compress:
          theList = theCompressedFilesList
              
        

    newArt = 0
    uploadPath = ''

# new article

    if options.newId:
        oldArt = lastArticle()
        newArt = oldArt+1
        print 'Creating new article with identifier '+str(newArt)+' ...\n'
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
            raise('Article '+str(newArt)+' to be updated does not exist!')

# list article
        
    elif options.artIdLi !=0:
        listPath = defaultEOSRootPath+'/'+str(options.artIdLi)
        theCommand = defaultEOSlistCommand+' '+listPath
        exeList = subprocess.Popen(["/bin/sh","-c",theCommand], stdout=subprocess.PIPE)
        for line in exeList.stdout.readlines():
            if findXrdDir(line) != None:
                print findXrdDir(line)


    if newArt > 0:
        fileUpload(uploadPath,theList, reallyDoIt)
        listPath = defaultEOSRootPath+'/'+str(newArt)
        print ''
        print 'Listing the '+str(newArt)+' article content after upload:'
        theCommand = defaultEOSlistCommand+' '+listPath
        if reallyDoIt:
          exeFullList = subprocess.Popen(["/bin/sh","-c",theCommand])
          exeFullList.communicate()
        else:
          print 'Dry run, nothing was done'
        
