#!/usr/bin/env python

from optparse import OptionParser

import sys,os, re, pprint
import castortools


def isDir( castorFile ):
    pattern = re.compile( '^(d).*' )

    if pattern.match( castorFile ):
#        print "is a directory"
        return True
    else:
        return False

    
def getFileName( line ):
    components = line.split()
#    print components
    return components[8]

def replicateDirStructure( castor1, castor2 ):

    print 'Entering ',castor2, '--------------------'
    
    files = os.popen('nsls -l ' + castor1)
    # os.system('nsls ' + castor1)
    
    for file in files.readlines():
        if isDir( file ): 
            fileName = getFileName( file )
            print 'creating', fileName

            newDir1 = castor1 + '/' + fileName
            newDir2 = castor2 + '/' + fileName
            os.system('rfmkdir '+ newDir2)
            replicateDirStructure( newDir1, newDir2 ) 
            
    

parser = OptionParser()
parser.usage = "%prog <source castor dir> <destination castor dir>: replicate the directory structure of the first castor dir in the second one."
parser.add_option("-n", "--negate", action="store_true",
                  dest="negate",
                  help="do not proceed",
                  default=False)


(options,args) = parser.parse_args()

if len(args)!=2:
    parser.print_help()
    sys.exit(1)

castorDir1 = args[0]
castorDir2 = args[1]

replicateDirStructure( castorDir1, castorDir2 ) 

sys.exit(1)

files = castortools.matchingFiles( castorDir, regexp )

if options.negate:
    print 'NOT removing ',  
    pprint.pprint(files)
else:
    if options.kill == False:
        pprint.pprint(files)
        trash = castortools.createSubDir( castorDir, 'Trash')
        castortools.move( trash, files )
    else:
        castortools.remove( files )
