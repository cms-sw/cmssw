import glob
import os 
import sys
import pprint 
from stat import *


VREF = 'V4'

def nEvents(file):
    # could implement something using edmFileUtil
    return 2000.

def fileSize(file):
    return os.stat(file)[ST_SIZE]/1024./nEvents(file)

def frac(num, denom):
    num = fileSize(num)
    denom = fileSize(denom)
    return num/denom * 100

def printFile(file,aodFile, v2File ):
    fracAod = frac( file, aodFile )
    fracVREF = frac( file, v2File )
    return '{file:<40} size/evt = {size:6.1f} kB  ({fracAod:5.1f}% AOD) ({fracVREF:5.1f}% VREF)'.format( file=file,
                                                                                                    size=fileSize(file),
                                                                                                    fracAod=fracAod,
                                                                                                    fracVREF=fracVREF)

def dumpCollections(file):
    colFile = os.path.splitext(file)[0] + '.txt'
    cmd = ['edmFileUtil -P',
           file,
           "| awk '{print $6}' >",
           colFile
           ]
    os.system( ' '.join( cmd ) )

    iFile = open( colFile )
    return set( [line.rstrip('\n') for line in iFile] )

def analyzeCollections(colSet, otherColSet, fileName ):
    added = colSet - otherColSet
    removed = otherColSet - colSet
    oFile = open( fileName, 'w')
    oFile.write('\nadded++++++++++++++++++++\n\n')
    for line in sorted(added):
        oFile.write( line )
        oFile.write('\n')
    oFile.write('\n\nremoved------------------\n\n')
    for line in sorted(removed):
        oFile.write( line )
        oFile.write('\n')

def isAOD(file):
    fields = os.path.splitext(file)[0].split('_')
    if len(fields) == 2:
        return True
    else:
        return False

def isVREF(file):
    fields = os.path.splitext(file)[0].split('_')
    if len(fields)<3:
        return False
    version = fields[2]
    if version == VREF:
        return True
    else:
        return False

def find( files, predicate ):
    theFile = None
    for file in files:
        isIt = predicate( file )
        if isIt:
            if theFile is None:
                theFile = file
            else:
                raise Exception('2 files look like it...', theFile, file)
    return theFile

def findAOD(files):
    return find( files, isAOD)


def findVREF(files):
    return find( files, isVREF)


def processDir(dir):
    baseDir = os.getcwd()
    os.chdir( dir )
    
    rootFiles = glob.glob( '*.root' )
    print rootFiles
    AOD = findAOD(rootFiles)
    aodCollections = dumpCollections( AOD )

    vref = findVREF(rootFiles)
    
    v2Collections = dumpCollections( vref )
    #     print AOD, aodSize

    oFileSizeReport = open('size_report.txt','w')
    oFileSizeReport.write('\nSIZE REPORT\n')
    oFileSizeReport.write('-----------\n\n')
    
    for file in sorted(rootFiles):
        if file is not AOD:
            collections = dumpCollections(file)
            oFile = os.path.splitext(file)[0] + '_vs_AOD.txt'
            analyzeCollections( collections,
                                aodCollections,
                                oFile )
            if file is not vref:
                oFileVREF = os.path.splitext(file)[0] + '_vs_{vref}.txt'.format(vref=VREF)
                analyzeCollections( collections,
                                    v2Collections,
                                    oFileVREF )
        
        report = printFile(file, AOD, vref)
        print report 
        oFileSizeReport.write( report + '\n')
    os.chdir( baseDir )

if __name__ == '__main__':

    dir = sys.argv[1]
    processDir(dir)
