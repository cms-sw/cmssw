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
parser.usage = "%prog <castor dir> <regexp pattern> <chunk size>: merge a set of CMSSW root files on castor."
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

matchingFiles = castortools.matchingFiles( castorDir, regexp, protocol='rfio', castor=True)

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
    processFiles( chunk )
        



