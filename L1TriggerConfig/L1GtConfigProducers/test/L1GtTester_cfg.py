from __future__ import print_function
# cfg file to test L1 GT records
#
# V M Ghete  2008 - 2010 - 2012

import FWCore.ParameterSet.Config as cms

# choose a valid global tag for the release you are using 
#
# 5_2_X
#useGlobalTag='GR_R_52_V9'
useGlobalTag='START52_V10'

# run number to retrieve the records - irrelevant if records are overwritten or
# the global tag is a MC global tag, with infinite IoV
useRunNumber = 194251

# print L1 GT prescale factors and trigger mask
printPrescaleFactorsAndMasks = True
#printPrescaleFactorsAndMasks = False

# print L1 GT board maps
printBoardMaps = True
#printBoardMaps = False

# print L1 GT stable parameters
printStableParameters = True
#printStableParameters = False

# print L1 GT parameters
printParameters = True
#printParameters = False

# print L1 GT PSB setup
printPsbSetup = True
#printPsbSetup = False

##########################################################################################

# process

processName = "L1GtTester"
process = cms.Process(processName)

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptyIOVSource",
                        timetype = cms.string('runnumber'),
                        firstValue = cms.uint64(useRunNumber),
                        lastValue = cms.uint64(useRunNumber),
                        interval = cms.uint64(1)
                        )


# import standard configurations, load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
# retrieve also the HLT menu

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'

# records to be printed

process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTester_cff')

# prescale factors and masks
#process.l1GtPrescaleFactorsAndMasksTester.TesterPrescaleFactors = True
#process.l1GtPrescaleFactorsAndMasksTester.TesterTriggerMask = True
#process.l1GtPrescaleFactorsAndMasksTester.TesterTriggerVetoMask = True
#process.l1GtPrescaleFactorsAndMasksTester.RetrieveInBeginRun = True
#process.l1GtPrescaleFactorsAndMasksTester.RetrieveInBeginLuminosityBlock = False
#process.l1GtPrescaleFactorsAndMasksTester.RetrieveInAnalyze = False
#process.l1GtPrescaleFactorsAndMasksTester.PrintInBeginRun = True
#process.l1GtPrescaleFactorsAndMasksTester.PrintInBeginLuminosityBlock = False
#process.l1GtPrescaleFactorsAndMasksTester.PrintInAnalyze = False
#process.l1GtPrescaleFactorsAndMasksTester.PrintOutput = 0

# Path definitions
process.pathL1GtStableParameters = cms.Path(process.seqL1GtStableParameters)
process.pathL1GtParameters = cms.Path(process.seqL1GtParameters)
process.pathL1GtBoardMaps = cms.Path(process.seqL1GtBoardMaps)
process.pathL1GtPsbSetup = cms.Path(process.seqL1GtPsbSetup)
process.pathL1GtPrescaleFactorsAndMasks = cms.Path(process.seqL1GtPrescaleFactorsAndMasks)

# Schedule definition
process.schedule = cms.Schedule()

print('')

if printStableParameters == True :
    process.schedule.extend([process.pathL1GtStableParameters])
    print("Printing L1 GT stable parameters from global tag ", useGlobalTag)
else :
    print("L1 GT stable parameters from ", useGlobalTag, " not requested to be printed")

if printParameters == True :
    process.schedule.extend([process.pathL1GtParameters])
    print("Printing L1 GT parameters from global tag ", useGlobalTag)
else :
    print("L1 GT parameters from ", useGlobalTag, " not requested to be printed")

if printBoardMaps == True :
    process.schedule.extend([process.pathL1GtBoardMaps])
    print("Printing L1 GT board maps from global tag ", useGlobalTag)
else :
    print("L1 GT board maps from ", useGlobalTag, " not requested to be printed")

if printPsbSetup == True :
    process.schedule.extend([process.pathL1GtPsbSetup])
    print("Printing L1 GT PSB setup from global tag ", useGlobalTag)
else :
    print("L1 GT PSB setup from ", useGlobalTag, " not requested to be printed")

if printPrescaleFactorsAndMasks == True :
    process.schedule.extend([process.pathL1GtPrescaleFactorsAndMasks])
    print("Printing L1 GT prescale factors and masks from global tag ", useGlobalTag)
else :
    print("L1 GT prescale factors and masks from ", useGlobalTag, " not requested to be printed")
    


# services

# Message Logger
process.MessageLogger.cerr.enable = False

process.MessageLogger.files.L1GtTester_errors = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtPrescaleFactorsAndMasksTester = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
       )

process.MessageLogger.files.L1GtTester_warnings = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtPrescaleFactorsAndMasksTester = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.files.L1GtTester_info = cms.untracked.PSet( 
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtPrescaleFactorsAndMasksTester = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.files.L1GtTester_debug = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtPrescaleFactorsAndMasksTester = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

