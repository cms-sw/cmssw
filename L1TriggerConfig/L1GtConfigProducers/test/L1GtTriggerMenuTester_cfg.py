#
# cfg file to print the L1 GT trigger menu using L1Trigger_custom 
#     options to choose the source of L1 Menu are to be given in customiseL1Menu method 
#     from L1Trigger_custom
#
# V M Ghete  2008 - 2010 - 2012


import FWCore.ParameterSet.Config as cms

# choose a valid global tag for the release you are using 
#
# 5_3_X
useGlobalTag='GR_P_V42B'

# run number to retrieve the menu - irrelevant if menu is overwritten or
# the global tag is a MC global tag, with infinite IoV
useRunNumber = 194251

# options to retrieve a HLT menu from HLTrigger.Configuration.HLT_GRun_cff
# and to print the HLT paths using the L1 triggers as seeds

# use the HLT menu given in this configuration 
#useHltMenuOption = True
useHltMenuOption = False

##########################################################################################

# process

processName = "L1GtTriggerMenuTester"
process = cms.Process(processName)

process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuTester_cfi")
process.l1GtTriggerMenuTester.OverwriteHtmlFile = True
process.l1GtTriggerMenuTester.HtmlFile = "L1Menu_Collisions2015_25ns_v1_L1T_Scales_20101224_Imp0_0x102f.html"
process.l1GtTriggerMenuTester.UseHltMenu = useHltMenuOption
process.l1GtTriggerMenuTester.HltProcessName = processName
#process.l1GtTriggerMenuTester.NoThrowIncompatibleMenu = False
#process.l1GtTriggerMenuTester.PrintPfsRates = True
#process.l1GtTriggerMenuTester.IndexPfSet = 1


from L1Trigger.Configuration.L1Trigger_custom import customiseL1Menu
process=customiseL1Menu(process)

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

if useHltMenuOption == True :
    readFiles = cms.untracked.vstring()
    secFiles = cms.untracked.vstring() 
    selectedLumis= cms.untracked.VLuminosityBlockRange()
    selectedEvents = cms.untracked.VEventRange()

    # files are used only to run HLT on an event - the HLT menu is overwritten with the menu from HLT_GRun_cff
    readFiles.extend( [
                 'file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root'
                ] );

    process.source = cms.Source ('PoolSource', 
                                 fileNames=readFiles, 
                                 secondaryFileNames=secFiles,
                                 lumisToProcess = selectedLumis,
                                 eventsToProcess = selectedEvents
                                 )
else :
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
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')

if useHltMenuOption == True :
    process.load('HLTrigger.Configuration.HLT_GRun_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.GlobalTag.globaltag = useGlobalTag+'::All'
# for GR_H_
#process.GlobalTag.connect = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG' 
#process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/') 

# path to be run

if useHltMenuOption == True :
    process.p = cms.Path(process.RawToDigi*process.l1GtTriggerMenuTester)
else :
    process.p = cms.Path(process.l1GtTriggerMenuTester)
    

# services

# Message Logger
process.MessageLogger.debugModules = ['l1GtTriggerMenuTester']
process.MessageLogger.categories.append('L1GtTriggerMenuTester')
process.MessageLogger.categories.append('L1GtTriggerMenuTesterWiki')
process.MessageLogger.destinations = ['L1GtTriggerMenuTester_errors', 
                                      'L1GtTriggerMenuTester_warnings', 
                                      'L1GtTriggerMenuTester_info', 
                                      'L1GtTriggerMenuTester_wiki', 
                                      'L1GtTriggerMenuTester_debug'
                                      ]
process.MessageLogger.statistics = []

process.MessageLogger.L1GtTriggerMenuTester_errors = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtTriggerMenuTester = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
       )

process.MessageLogger.L1GtTriggerMenuTester_warnings = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtTriggerMenuTester = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.L1GtTriggerMenuTester_info = cms.untracked.PSet( 
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtTriggerMenuTester = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.L1GtTriggerMenuTester_wiki = cms.untracked.PSet( 
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtTriggerMenuTesterWiki = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.L1GtTriggerMenuTester_debug = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtTriggerMenuTester = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )
