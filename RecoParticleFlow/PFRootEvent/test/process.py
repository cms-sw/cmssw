#!/usr/bin/env python

import sys, string, os, re
from optparse import OptionParser


# replace a given line by another one
def replaceLine( line, tag1, tag2, value ):

    comment = '^\s*\/\/'
    pattern = '(\s*%s\s+%s\s+)' % (tag1, tag2)

    #print 'pattern ', pattern

    # commented line    
    p = re.compile(comment);
    if p.match(line):
        return line


    p = re.compile(pattern);
    m=p.match(line)
    #print line
    if m:
        line = '%s %s %s\n' % (tag1, tag2, value)
        #print 'matched',line
    return line


def createOptFile( sourceFileName, newOptFileName, tag1, tag2, value ):


    if( os.path.isfile(sourceFileName)=='' ):
        print 'file ',sourceFileName,' does not exist'
        sys.exit(5)

    ifile=open(sourceFileName,'r')


    ofile=open(newOptFileName,'w')

    if ofile==None:
        print 'cannot open ',ofileName
        sys.exit(5)

    
    (root,ext) = os.path.splitext(newOptFileName)
    rootFileName = '%s.root'% (root)
    for line in ifile.readlines():
        newLine = replaceLine(line, 'root','outfile',rootFileName)
        newLine = replaceLine(newLine, tag1,tag2,value)
        ofile.write(newLine)



def getOptFileName( macroFile, tag1, tag2, value ):

    if( os.path.isfile(macroFile)=='' ):
        print 'file ',macroFile,' does not exist'
        sys.exit(5)

    comment = '^\s*\/\/'

    pcom = re.compile(comment);
    p = re.compile('(\s*PFRootEvent\S*\s+\S+\(\")(\S+)(\"\);)')

    ifile=open(macroFile,'r')

    macroDirName = os.path.dirname(macroFile)
    macroBaseName = os.path.basename(macroFile)
    (root,ext) = os.path.splitext(macroBaseName)
    
    newMacroFileName = '%s/ppy_%s_%s_%s_%s.C'% (macroDirName, root, tag1, tag2, value)
    ofile=open(newMacroFileName,'w')
    if ofile==None:
        print 'cannot open ',newMacroFileName
        sys.exit(5)

    newOptFileName = None
    for line in ifile.readlines():
        mcom = pcom.match(line)
        if mcom!=None:    # commented line    
            continue
        
        m=p.match( line )
        if m!= None:
            sourceOptFileName = m.group(2)
            (root,ext) = os.path.splitext(sourceOptFileName)
            newOptFileName = 'ppy_%s_%s_%s_%s.opt' % (root,tag1, tag2, value)
            line = '%s%s%s' % (m.group(1), newOptFileName, m.group(3))
        ofile.write(line)

    return sourceOptFileName, newOptFileName, newMacroFileName

 

usage = "usage: %prog [options] macro tag1 tag2 value"
#parser = OptionParser(usage=usage)
#(options,args) = parser.parse_args()


#print sys.argv
args = sys.argv
if len(args)!=5:
    # parser.print_help()
    print usage
    sys.exit(1)

macroFile = args[1]
tag1 = args[2]
tag2 = args[3]
value = args[4]

# get optfile from the macro
(sourceOptFileName, newOptFileName, newMacro) = getOptFileName(macroFile, tag1, tag2, value)

# create optfile with new values for tag1/tag2
createOptFile( sourceOptFileName, newOptFileName, tag1, tag2, value )


# run
cmd = 'root -b %s' % newMacro
os.system( cmd )



