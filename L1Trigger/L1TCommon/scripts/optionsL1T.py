#
# L1T Command Line Options:
#
# Append this snippet to you cmsDriver.py config file like this:
#
#   cat L1Trigger/L1TCommon/scripts/optionsL1T.py
#
# to provide support for command-line options such as:
#
#    maxEvents=<n>
#    skip=<n>
#    ntuple=<file>
#    inputFiles="file1.root,file2.root" 
#
#    menu (not yet implemented)
#
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os
import sys
import commands
options = VarParsing.VarParsing ('analysis')
options = VarParsing.VarParsing ('analysis')
options.register ('ntuple', "",  VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,  "The output ntuple file name")
options.register ('menu',   "",  VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,  "Override the L1 menu with specified XML file")
options.register ('skip',   "",  VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,     "The number of events to skip initially")
options.ntuple = ''
options.menu = ''
options.skip = 0 
print options

options.parseArguments()

if (not options.menu == ""):
    print "L1T INFO:  menu override in command line not yet implemented..."

if (not options.ntuple == ""):
    print "L1T INFO:  using command line option ntuple:  ", options.ntuple
    if (hasattr(process,"TFileService")):
        process.TFileService.fileName = cms.string(options.ntuple)
        #print process.TFileService

if (hasattr(process,"maxEvents")):
    print "L1T INFO:  using command line option maxEvents:  ", options.maxEvents
    process.maxEvents.input = options.maxEvents
    #print process.maxEvents

if (hasattr(process,"source")):
    if options.skip > 0:
        print "L1T INFO:  using command line option skip:  ", options.skip
        process.source.skipEvents = cms.untracked.uint32(options.skip)
    if (not options.inputFiles == []):
        print "L1T INFO:  using command line option inputFiles:  ", options.inputFiles
        process.source.fileNames = cms.untracked.vstring(options.inputFiles)
