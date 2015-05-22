#!/usr/bin/env python
# Colin Bernet, Dec 2009

import FWCore.ParameterSet.Config as cms
from optparse import OptionParser
import os,sys

def outputFile( args ):
    return args[0]

def inputFiles( args ):
    return args[1:]

def protocol( files ):
    newFiles = []
    for file in files:
        file = 'file:' + file
        newFiles.append(file)
    return newFiles

def testRoot( files ):
    for file in files:
        if not os.path.isfile( file ):
            print 'file', file, 'does not exist'
            return False
        (dummy, ext) = os.path.splitext( file )
        if not ext == '.root':
            print 'file', file, 'is not a root file'
            return False
    return True


parser = OptionParser()
parser.usage = "%prog <output file> <input files>\nMerge EDM files"
parser.add_option("-n", "--negate", action="store_true",
                  dest="negate", default=False,
                  help="create cfg file, but do not cmsRun")

(options,args) = parser.parse_args()

if len(args) < 2:
    parser.print_help()
    sys.exit(1)

outFile =  outputFile( args )
inFiles = inputFiles( args )

if not testRoot( inFiles ): sys.exit(1)

filesWithProtocol = protocol( inFiles )


process = cms.Process("COPY")

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
      filesWithProtocol
    ),
    )

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.EventContent.EventContent_cff")
process.out = cms.OutputModule(
    "PoolOutputModule",
    # process.AODSIMEventContent,
    outputCommands =  cms.untracked.vstring(
      'keep *'
      ),
    fileName = cms.untracked.string( outFile ),
    )

process.endpath = cms.EndPath(
    process.out
    )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

outpyFile = open("tmpConfig.py","w")
outpyFile.write("import FWCore.ParameterSet.Config as cms\n")
outpyFile.write(process.dumpPython())
outpyFile.close()

print process.source.fileNames
print 'will be merged into ', outFile

if not options.negate:
    os.system("cmsRun tmpConfig.py")

