#! /usr/bin/env python

__version__ = "$Revision: 1.5 $"

import os
import subprocess
import time
import re

defaultEOSinitCommand = 'source /afs/cern.ch/cms/caf/eos.sh ; alias'
defaultEOSRootPath = '/eos/cms/'
defaultEOSLoadPath = '/store/lhe'
defaultEOSBasePath = defaultEOSRootPath+defaultEOSLoadPath
defaultEOSlistCommand = 'eoscms ls'
defaultEOSfulllistCommand = 'eoscms ls -l'
defaultEOSmkdirCommand = 'eoscms mkdir'
defaultEOScpCommand = 'cmsStage'

def articleExist(artId):

    itExists = False
    uploadPath = defaultEOSBasePath
    theCommand = EOSCommandPath+defaultEOSlistCommand+' '+uploadPath
    dirList = subprocess.Popen(["/bin/sh","-c",theCommand], stdout=subprocess.PIPE)
    for line in dirList.stdout.readlines():
        if line.rstrip('\n') == str(artId): 
            itExists = True

    return itExists

def lastArticle():

    artList = [0]

    theCommand = EOSCommandPath+defaultEOSlistCommand+' '+defaultEOSBasePath
    dirList = subprocess.Popen(["/bin/sh","-c",theCommand], stdout=subprocess.PIPE)
    for line in dirList.stdout.readlines():
        try:
            artList.append(int(line.rstrip('\n')))
        except:
            break

    return max(artList)


def fileUpload(uploadPath,lheList, reallyDoIt):

    inUploadScript = ''

    for f in lheList:
        # Check the file existence
        newFileName = uploadPath+'/'+str(f)
        addFile = True
        additionalOption = ''  
        print newFileName
        theCommand = EOSCommandPath+defaultEOSfulllistCommand+' '+defaultEOSRootPath+newFileName+' &> /dev/null' 
        exeFullList = subprocess.Popen(["/bin/sh","-c",theCommand])
        result = exeFullList.wait()
        if result == 0:
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
            inUploadScript += defaultEOScpCommand+additionalOption+' '+str(f)+' '+uploadPath+'\n'

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
        for f in theList: 
            # Check the file name extension
            if not f.lower().endswith(".lhe"):
                raise Exception('Input file name must have the "lhe" final extension!')
            # Check the local file existence
            if not os.path.exists(f):
                raise Exception('Input file '+f+' does not exists')
        

    newArt = 0
    uploadPath = ''

    loadEOSpath = subprocess.Popen(["/bin/sh","-c",defaultEOSinitCommand], stdout=subprocess.PIPE)
    EOSCommandPath = ' ' 
    for line in loadEOSpath.stdout.readlines():
        EOSCommandPath =  line.rsplit('\'')[1].rstrip('eoscms')

# new article

    if options.newId:
        oldArt = lastArticle()
        newArt = oldArt+1
        print 'Creating new article with identifier '+str(newArt)+' ...\n'
        uploadPath = defaultEOSBasePath+'/'+str(newArt)
        theCommand = EOSCommandPath+defaultEOSmkdirCommand+' '+uploadPath
        if reallyDoIt:
          exeUpload = subprocess.Popen(["/bin/sh","-c",theCommand])
          exeUpload.communicate()
        uploadPath = defaultEOSLoadPath+'/'+str(newArt)

# update article
        
    elif options.artIdUp != 0:
        newArt = options.artIdUp
        if articleExist(newArt):
            uploadPath = defaultEOSLoadPath+'/'+str(newArt)
        else:
            raise('Article '+newArt+' to be updated does not exist!')

# list article
        
    elif options.artIdLi !=0:
        listPath = defaultEOSBasePath+'/'+str(options.artIdLi)
        theCommand = EOSCommandPath+defaultEOSfulllistCommand+' '+listPath
        exeList = subprocess.Popen(["/bin/sh","-c",theCommand])
        exeList.communicate()


    if newArt > 0:
        fileUpload(uploadPath,theList, reallyDoIt)
        listPath = defaultEOSBasePath+'/'+str(newArt)
        print ''
        print 'Listing the '+str(newArt)+' article content after upload:'
        theCommand = EOSCommandPath+defaultEOSfulllistCommand+' '+listPath
        if reallyDoIt:
          exeFullList = subprocess.Popen(["/bin/sh","-c",theCommand])
          exeFullList.communicate()
        else:
          print 'Dry run, nothing was done'
        
