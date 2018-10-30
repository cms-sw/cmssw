from __future__ import print_function
import os

import FWCore.ParameterSet.Config as cms




##
## Setup command line options
##
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
options = VarParsing.VarParsing ('standard')
options.register('iterNumber', 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Iteration number")
options.register('setBaseline', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Set baseline")
options.register('measurementName', "workingArea", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Folder in which to store results")

# get and parse the command line arguments
options.parseArguments()

print("Iteration number: ", options.iterNumber)
print("Set baseline: ", options.setBaseline)



if options.setBaseline:
    print("Set baseline mode, do not create APE DB-object")
    exit(0)


##
## Process definition
##
process = cms.Process("APE")
# we need conditions

#;;;;;;;;;;;;;;;new line;;;;;;;;;;;;;;;
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag

process.load("Configuration.Geometry.GeometryRecoDB_cff")

# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_design', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_design', '')


# Alignment producer
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")
from Alignment.CommonAlignmentAlgorithm.ApeSettingAlgorithm_cfi import *
process.AlignmentProducer.algoConfig = ApeSettingAlgorithm
process.AlignmentProducer.saveApeToDB = True
process.AlignmentProducer.algoConfig.readApeFromASCII = True
process.AlignmentProducer.algoConfig.setComposites = False
process.AlignmentProducer.algoConfig.readLocalNotGlobal = True
# CAVEAT: Input file name has to start with a Package name...
process.AlignmentProducer.algoConfig.apeASCIIReadFile = 'Alignment/APEEstimation/hists/'+options.measurementName+'/iter'+str(options.iterNumber)+'/allData_apeOutput.txt'
process.AlignmentProducer.algoConfig.saveApeToASCII = False
process.AlignmentProducer.algoConfig.saveComposites = False
process.AlignmentProducer.algoConfig.apeASCIISaveFile = 'myLocalDump.txt'
        

# to be refined...
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring('cout', 'alignment'),
    categories = cms.untracked.vstring('Alignment'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
    ),
    alignment = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('INFO'),
        Alignment = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    destinations = cms.untracked.vstring('cout',  ## .log automatically
        'alignment')
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

from CondCore.CondDB.CondDB_cfi import *
CondDBAlignmentError = CondDB.clone(connect = cms.string('sqlite_file:'+os.environ['CMSSW_BASE']+'/src/Alignment/APEEstimation/hists/'+options.measurementName+'/apeObjects/apeIter'+str(options.iterNumber)+'.db'))
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    CondDBAlignmentError,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
        record = cms.string('TrackerAlignmentErrorExtendedRcd'),
            tag = cms.string('APEs')
        )
    )
)



