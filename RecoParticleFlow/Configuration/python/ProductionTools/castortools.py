#!/usr/bin/env python
# API for castor 
# Colin Bernet, July 2009 

from optparse import OptionParser
import sys,os, re, pprint


def isCastorDir( dir ):
    pattern = re.compile( '^/castor' )
    if pattern.match( dir ):
        return True
    else:
        return False
    

# returns all files in a directory matching regexp.
# the directory can be a castor dir.
# optionnally, the protocol (rfio: or file:) is prepended to the absolute
# file name
def matchingFiles( dir, regexp, protocol=None, castor=True):

    ls = 'rfdir'
 
    try:
        pattern = re.compile( regexp )
    except:
        print 'please enter a valid regular expression '
        sys.exit(1)

    allFiles = os.popen("%s %s | awk '{print $9}'" % (ls, dir))

    matchingFiles = []
    for file in allFiles.readlines():
        file = file.rstrip()
        
        m = pattern.match( file )
        if m:
            fullCastorFile = '%s/%s' % (dir, file)
            if protocol:
                fullCastorFile = '%s%s/%s' % (protocol, dir, file)
            matchingFiles.append( fullCastorFile )

    allFiles.close()

    return matchingFiles

# does not work
def rootEventNumber( file ):
    tmp = open("nEvents.C", "w")
    tmp.write('''
void nEvents(const char* f) {
  loadFWLite();
  TFile* fp = TFile::Open(f);
  cout<<Events->GetEntries()<<endl;
  gApplication->Terminate();
}
''')
    command = 'root -b \'nEvents.C(\"%s\")\' ' % file
    print command
    output = os.popen('root -b \'nEvents.C(\"%s\")\' ' % file)
    print 'done'
    output.close()
    print 'closed'

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
        rfrm = 'rfrm %s' % file
        print rfrm
        os.system( rfrm )

# copy a set of files to a castor directory
def cp( absDestDir, files ):
    cp = 'cp'
    destIsCastorDir = isCastorDir(absDestDir)
    if destIsCastorDir: 
        cp = 'rfcp'
        createCastorDir( absDestDir )
        
    for file in files:

        if destIsCastorDir == False:
            if isCastorDir( os.path.abspath(file) ):
                cp = 'rfcp'
            else:
                cp = 'cp'
        
        cpfile = '%s %s %s' % (cp, file,absDestDir)
        print cpfile
        os.system(cpfile)
        
        


