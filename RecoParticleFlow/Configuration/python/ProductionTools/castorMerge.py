#!/usr/bin/env python

from optparse import OptionParser
import sys,os, re, pprint
import castortools 
import FWCore.ParameterSet.Config as cms

chunkNumber = 0

def processFiles( regexp, files ):
    
    global chunkNumber

    if len(files) == 0:
        print 'processFiles: no file in input'
        sys.exit(2)

    print 'Processing files:'
    pprint.pprint( files )
    
    process = cms.Process("COPY")

    process.source = cms.Source(
        "PoolSource",
        fileNames = cms.untracked.vstring( files ),
        noEventSort = cms.untracked.bool(True),
        duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
        )
    
    # build output file name
    file = os.path.basename( files[0] )
    (prefix, index) = castortools.filePrefixAndIndex( regexp, file)
    
    tmpRootFile = '/tmp/%s_chunk%d.root' % (prefix,chunkNumber)

    print '  destination: ', tmpRootFile
    process.aod = cms.OutputModule(
        "PoolOutputModule",
        fileName = cms.untracked.string( tmpRootFile ),
        outputCommands = cms.untracked.vstring( 'keep *' )
        )
    

    process.outpath = cms.EndPath(process.aod)

    outFile = open("tmpConfig.py","w")
    outFile.write("import FWCore.ParameterSet.Config as cms\n")
    outFile.write(process.dumpPython())
    outFile.close()

    chunkNumber = chunkNumber+1

    if options.negate == True:
        return

    chunkDir = castortools.createSubDir( castorDir, 'Chunks' )
    
    os.system("cmsRun tmpConfig.py")
    print 'done.'
    rfcp = "rfcp %s %s" % (tmpRootFile, chunkDir)
    print rfcp,'...'
    os.system( rfcp )
    os.system("rm %s" % tmpRootFile)
    print 'temporary files removed.'

    
parser = OptionParser()
parser.usage = "%prog <castor dir> <regexp pattern> <chunk size>: merge a set of CMSSW root files on castor. Temporary merged files are created on /tmp of the local machine, and are then migrated to a Chunk/ subdirectory of your input castor directory. Therefore, you need write access to the input castor directory.\n\nThe regexp pattern should contain 2 statements in parenthesis: the first one should match the file prefix, and the second one the file number. The name of the merged file will start by the file prefix. The file number is used to decide which files to take in input. The chunk size is the number of input files to be merged in a given output file.\n\nExample (just try. the -n option negates the command!):\ncastorMerge.py  /castor/cern.ch/user/c/cbern/CMSSW312/SinglePions '(.*)_(\d+)\.root' 2 -n"
parser.add_option("-n", "--negate", action="store_true",
                  dest="negate",
                  help="do not produce the merged files",
                  default=False)



(options,args) = parser.parse_args()

if len(args)!=3:
    parser.print_help()
    sys.exit(1)

castorDir = args[0]
regexp = args[1]
chunkSize = int(args[2])

print 'Merging files in: ', castorDir

matchingFiles = castortools.matchingFiles( castorDir, regexp, protocol='rfio:', castor=True)

# grouping files
count = 0
chunk = []
for file in matchingFiles:
    count += 1
    chunk.append( file )
    if count == chunkSize:
        count = 0
        processFiles( regexp, chunk )
        chunk = []
        
# remaining files:
if len(chunk)>0:
    processFiles( regexp, chunk )
        



