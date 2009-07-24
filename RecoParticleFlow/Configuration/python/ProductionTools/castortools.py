#!/usr/bin/env python

from optparse import OptionParser
import sys,os, re, pprint

# returns all castor files in a directory matching regexp
def allCastorFiles( castorDir, regexp ):

    try:
        pattern = re.compile( regexp )
    except:
        print 'please enter a valid regular expression '
        sys.exit(1)

    allFiles = os.popen("rfdir %s | awk '{print $9}'" % (castorDir))

    matchingFiles = []
    for file in allFiles.readlines():
        file = file.rstrip()
        
        m = pattern.match( file )
        if m:
            fullCastorFile = 'rfio:%s/%s' % (castorDir, file)
            matchingFiles.append( fullCastorFile )

    allFiles.close()

    return matchingFiles

# cleanup files with a size that is too small, out of a given tolerance. 
def cleanFiles( castorDir, regexp, tolerance):

    try:
        pattern = re.compile( regexp )
    except:
        print 'please enter a valid regular expression '
        sys.exit(1)

    allFiles = os.popen("rfdir %s | awk '{print $9}'" % (castorDir))
    sizes = os.popen("rfdir %s | awk '{print $5}'" % (castorDir))

    averageSize = 0
    count = 0.

    matchingFiles = []
    print 'Matching files: '
    for file,size in zip( allFiles.readlines(), sizes.readlines()):
        file = file.rstrip()
        size = float(size.rstrip())

        m = pattern.match( file )
        if m:
            print file
            fullCastorFile = '%s/%s' % (castorDir, file)
            matchingFiles.append( (fullCastorFile, size) )
            averageSize += size
            count += 1

    averageSize /= count
    print 'average file size = ',averageSize

    cleanFiles = []
    dirtyFiles = []

    for file, size in matchingFiles:
        relDiff = (averageSize - size) / averageSize
        if relDiff < tolerance:
            # ok
            # print file, size, relDiff
            cleanFiles.append( file )
        else:
            print 'skipping', file, ': size too small: ', size, relDiff
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
        return int(m.group(1))
    else:
        print file, ': cannot find number.'
        return -1

# extract the file index using regexp
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
def sync( regexp, files1, files2):

    # should be defined from outside
    regexp = '_(\d+)\.root'

    numAndFile1 = extractNumberAndSort( regexp, files1 )
    numAndFile2 = extractNumberAndSort( regexp, files2 )
   
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
    out = os.system( 'rfdir %s' % absName )
    print out
    if out!=0:
        # dir does not exist
        os.system( 'rfmkdir %s' % absName )
    return absName

# move a set of files to another directory
def move( absDestDir, files ):
    for file in files:
        baseName = os.path.basename(file)
        rfrename = 'rfrename %s %s/%s' % (file, absDestDir, baseName)
        #print rfrename
        os.system( rfrename )
        

