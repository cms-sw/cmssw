#
# cfg file to print the L1 GT trigger menu using L1Trigger_custom 
# options to choose the source of L1 Menu are to be given in L1Trigger_custom
#
# V M Ghete  2008 - 2010 - 


import FWCore.ParameterSet.Config as cms

# choose a valid global tag for the release you are using 
# for the option "l1MenuSource='globalTag'", the menu from global tag will be printed  
#
# 3_8_X gTags
useGlobalTag='TESTL1_ST311'

# process
process = cms.Process("L1GtTriggerMenuTest")
process.l1GtTriggerMenuTest = cms.EDAnalyzer("L1GtTriggerMenuTester")

from L1Trigger.Configuration.L1Trigger_custom import customiseL1Menu
process=customiseL1Menu(process)

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'

# path to be run
process.p = cms.Path(process.l1GtTriggerMenuTest)

# services

# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules = ['l1GtTriggerMenuTest']
process.MessageLogger.cout = cms.untracked.PSet(
    #INFO = cms.untracked.PSet(
    #    limit = cms.untracked.int32(-1)
    #)#,
    threshold = cms.untracked.string('ERROR'), ## DEBUG 
    
    ERROR = cms.untracked.PSet( ## DEBUG, all messages  

        limit = cms.untracked.int32(-1)
    )
)


