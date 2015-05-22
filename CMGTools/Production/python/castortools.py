#!/usr/bin/env python
# API for castor 
# Colin Bernet, July 2009 

from optparse import OptionParser
import sys,os, re, pprint
import subprocess


def isCastorDir( dir ):
    pattern = re.compile( '^/castor' )
    if pattern.match( dir ):
        return True
    else:
        return False

#COLIN is it still in use?
# yes, in crabProd.py. could think about removing it
def isCastorFile( file ):
    if isLFN(file):
        file = lfnToCastor(file)
    os.system( 'nsls ' + file )
    ret = subprocess.call( ['nsls',file] )
    return not ret


def fileExists( file ):
    if isLFN(file):
        file =  lfnToCastor(file)
    castor = isCastorDir(file)
    ls = 'ls'
    if castor:
        ls = 'nsls'
    # ret = subprocess.call( [ls, file] )
    child = subprocess.Popen( [ls, file], stdout=subprocess.PIPE)
    child.communicate()
    # print ls, file, child.returncode
    return not child.returncode

# returns all files in a directory matching regexp.
# the directory can be a castor dir.
# optionnally, the protocol (rfio: or file:) is prepended to the absolute
# file name
#COLIN: now that we are using LFNs, one should remove the castor argument (I guess)
def matchingFiles( dir, regexp, addProtocol=False, LFN=True):

    ls = 'nsls'

    localFiles = False
    if isLFN( dir ):
        dir = lfnToCastor( dir )
    elif not isCastorDir( dir ):
        # neither LFN nor castor file -> local file
        ls = 'ls'
        localFiles = True
        dir = os.getcwd() + '/' + dir
        
    try:
        pattern = re.compile( regexp )
    except:
        print 'please enter a valid regular expression '
        sys.exit(1)

    # allFiles = None
    # if localFiles:
    #    cmd = "%s %s/%s" % (ls, os.getcwd(), dir)

    cmd = "%s %s" % (ls, dir) 
    allFiles = os.popen(cmd)

    matchingFiles = []
    for file in allFiles.readlines():
        file = file.rstrip()
        
        m = pattern.match( file )
        if m:
            fullFileName = '%s/%s' % (dir, file)
            if addProtocol and localFiles:
                fullFileName = 'file:%s/%s' % ( dir, file)
            if not localFiles and LFN:
                fullFileName = fullFileName.replace( '/castor/cern.ch/cms/store', '/store')
            matchingFiles.append( fullFileName )

    allFiles.close()

    return matchingFiles


# returns the number of events in a file 
def numberOfEvents( file, castor=True):

    cmd = 'edmFileUtil -f rfio:%s' % file
    if castor==False:
        cmd = 'edmFileUtil -f file:%s' % file
        
    output = os.popen(cmd)
        
    pattern = re.compile( '\( (\d+) events,' )

    for line in output.readlines():
        m = pattern.search( line )
        if m:
            return int(m.group(1))

def emptyFiles( dir, regexp, castor=True):
    allFiles = matchingFiles( dir, regexp)
    emptyFiles = []
    for file in allFiles:
        print 'file ',file
        num = numberOfEvents(file, castor)
        print 'nEvents = ', num
        if num==0:
            emptyFiles.append( file )
    return emptyFiles 

# cleanup files with a size that is too small, out of a given tolerance. 
def cleanFiles( castorDir, regexp, tolerance = 999999.):

    try:
        pattern = re.compile( regexp )
    except:
        print 'please enter a valid regular expression '
        sys.exit(1)

    allFiles = os.popen("rfdir %s | awk '{print $9}'" % (castorDir))
    sizes = os.popen("rfdir %s | awk '{print $5}'" % (castorDir))

    averageSize = 0.
    count = 0.

    matchingFiles = []
    print 'Matching files: '
    for file,size in zip( allFiles.readlines(), sizes.readlines()):
        file = file.rstrip()
        fsize = float(size.rstrip())

        m = pattern.match( file )
        if m:
            print file, fsize
            fullCastorFile = '%s/%s' % (castorDir, file)
            matchingFiles.append( (fullCastorFile, fsize) )
            averageSize += fsize
            count += 1

    if count==0:
        print "none. check your regexps!"
        sys.exit(2)
    
    averageSize /= count
    print 'average file size = ',averageSize

    cleanFiles = []
    dirtyFiles = []

    for file, fsize in matchingFiles:
        relDiff = (averageSize - fsize) / averageSize
        if relDiff<float(tolerance):
            # ok
            print file, fsize, relDiff
            cleanFiles.append( file )
        else:
            print 'skipping', file, ': size too small: ', fsize, relDiff
            dirtyFiles.append( file )

    return (cleanFiles, dirtyFiles)



# returns the file index as an integer, found using the provided regexp 
def fileIndex( regexp, file ):

    try:
        numPattern = re.compile( regexp )
    except:
        print 'fileIndex: please enter a valid regular expression '
        sys.exit(1)

    m = numPattern.search( file )
    if m:
        try:
            return int( m.group(1) )
        except:
            print 'fileIndex: please modify your regular expression to find the file index. The expression should contain the string (\d+).'
            sys.exit(2)
            
    else:
        print file, "does not match your regexp ", regexp
        return -1

def filePrefixAndIndex( regexp, file ):

    try:
        pattern = re.compile( regexp )
    except:
        print 'fileIndex: please enter a valid regular expression '
        sys.exit(1)
        
    m = pattern.search( file )
    if m:
        try:
            return (m.group(1), int( m.group(2) ) )
        except:
            print 'fileIndex: please modify your regular expression to find the prefix and file index. The expression should contain 2 statements in parenthesis. the second one should be (\d+).'
            sys.exit(2)
            
    else:
        print file, "does not match your regexp ", regexp
        return -1
    

# extract the file index AND THE PREFIX using regexp
# sort both collections of files according to the index 
def extractNumberAndSort( regexp, files ):
    numAndFile = []
    for file in files:
        num = fileIndex( regexp, file )
        if num>-1:
            numAndFile.append( (num, file) )

    numAndFile.sort()

    return numAndFile 


# finds the file index using regexp
# sort both collections of files according to the index
# returns the list of single files in each collection 
def sync( regexp1, files1, regexp2, files2):

    # should be defined from outside
    numAndFile1 = extractNumberAndSort( regexp1, files1 )
    numAndFile2 = extractNumberAndSort( regexp2, files2 )
   
    i1 = 0
    i2 = 0

    single = [] 
    while i1<len(numAndFile1) and i2<len(numAndFile2):

        (n1, f1) = numAndFile1[i1]
        (n2, f2) = numAndFile2[i2]

        #        print f1
        #        print f2
        # print 'nums: ', n1, n2, 'index: ', i1, i2
        
        if n1<n2:
            print 'single: ', f1
            single.append(f1)
            i1 += 1
        elif n2<n1:
                print 'single: ', f2
                single.append(f2)
                i2 += 1
        else:
            i1 += 1
            i2 += 1
    return single

# create a subdirectory in an existing directory 
def createSubDir( castorDir, subDir ):
    absName = '%s/%s' % (castorDir, subDir)
    return createCastorDir( absName )


# create castor directory, if it does not already exist
def createCastorDir( absName ):
    out = os.system( 'rfdir %s' % absName )
    if out!=0:
        # dir does not exist
        os.system( 'rfmkdir %s' % absName )
    return absName
    

# move a set of files to another directory on castor
def move( absDestDir, files ):
    for file in files:
        baseName = os.path.basename(file)
        rfrename = 'rfrename %s %s/%s' % (file, absDestDir, baseName)
        print rfrename
        os.system( rfrename )

# remove a set of files
def remove( files ):
    for file in files:
        if isLFN( file ):
            file = lfnToCastor( file )
        rfrm = 'rfrm %s' % file
        print rfrm
        os.system( rfrm )

def protectedRemove( *args ):
    files = matchingFiles( *args )
    if len(files) == 0:
        return True 

    pprint.pprint( files )
    yesno = ''
    while yesno!='y' and yesno!='n':
        yesno = raw_input('Are you sure you want to remove these files [y/n]? ')
    if yesno == 'y':
        remove( files )
        print 'files removed'
        return True
    else:
        print 'cancelled'
        return False

def isLFN( file ):
    storePattern = re.compile('^/store.*')
    if storePattern.match( file ):
        return True
    else:
        return False

def lfnToCastor( file ):
    if isCastorDir(file):
        return file
    elif isLFN( file ):
        return '/castor/cern.ch/cms' + file
    else:
        raise NameError(file)

def castorToLFN( file ):
    return file.replace('/castor/cern.ch/cms','')


def cmsStage( absDestDir, files, force):

    destIsCastorDir = isCastorDir(absDestDir)
    if destIsCastorDir: 
        createCastorDir( absDestDir )

    for file in files:
        storefile = file.replace('/castor/cern.ch/cms','')
        forceOpt = ''
        if force:
            forceOpt = '-f'
        cmsStage = 'cmsStage %s %s %s ' % (forceOpt, storefile, absDestDir) 
        print cmsStage
        os.system( cmsStage )
        

def xrdcp( absDestDir, files ):
    cp = 'cp'
    destIsCastorDir = isCastorDir(absDestDir)
    if destIsCastorDir: 
        cp = 'xrdcp'
        createCastorDir( absDestDir )
        
    for file in files:

        cpfile = '%s %s %s' % (cp, file,absDestDir)
        
        if destIsCastorDir == False:
            if isCastorDir( os.path.abspath(file) ):
                cp = 'xrdcp'
                cpfile = '%s "root://castorcms/%s?svcClass=cmst3&stageHost=castorcms" %s' % (cp, file,absDestDir)

        print cpfile
        os.system(cpfile)

def cat(lfn):
    name = lfnToCastor(lfn)
    if not fileExists(name):
        raise Exception("File '%s' does not exist. Cannot do a cat." % name)
    output = subprocess.Popen(['rfcat',name], stdout=subprocess.PIPE).communicate()[0]
    return output

def stageHost():
    """Returns the CASTOR instance to use"""
    return os.environ.get('STAGE_HOST','castorcms')
        
def listFiles(dir, rec = False):
    """Recursively list a file or directory on castor"""
    cmd = 'dirlist'
    if rec:
        cmd = 'dirlistrec'
    files = subprocess.Popen(['xrd',stageHost(),cmd, dir], stdout=subprocess.PIPE).communicate()[0]
    result = []
    for f in files.split('\n'):
        s = f.split()
        if s: result.append(tuple(s))
    return result

