#
# L1 Pattern Generator
# ====================
#
# Generate a pattern file from RAW data by
#  - extracting the GT inputs from the GCT and GT readout
#  - Re-running the GT emulator with a different trigger menu
# 

import FWCore.ParameterSet.Config as cms
import sys

# choose one (and only one) of the following source
# usually the global tag must be used

# data or RelVal
useRelValSample = True
useRelValSample = False

# Frontier GlobalTag to use for EventSetup, should match source data

if useRelValSample == True :
    useGlobalTag = "START42_V14A"
else :
    useGlobalTag = "GR_P_V22"
    

#l1MenuSource='globalTag'
l1MenuSource='sqlFile'
#l1MenuSource='xmlFile'


if l1MenuSource == 'sqlFile' :
    # the menu will be read from the SQL file instead of the global tag
    
    
    # pp menu
    #useSqlFile = '/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2011_v6/sqlFile/L1Menu_Collisions2011_v6_mc.db'
    #menuDbTag = 'L1GtTriggerMenu_L1Menu_Collisions2011_v6_mc'

    
    # HI menu
    useSqlFile = '/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_CollisionsHeavyIons2011_v0/sqlFile/L1Menu_CollisionsHeavyIons2011_v0_mc.db'
    menuDbTag = 'L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc'

elif l1MenuSource == 'xmlFile' :
    # explicit choice of the L1 menu
    # un-comment the corresponding menu in the list of the menus
    triggerMenuXml = 'L1Menu_Commissioning2010_v6.xml'
    print '   Using', triggerMenuXml, ' file to overwrite the L1 menu from the global tag'    

else :
    print '   Using default L1 trigger menu from Global Tag ', useGlobalTag    

# Input files
if useRelValSample == True :
    dataFiles = [
        '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/1C7012EE-6FCB-E011-8B5A-0026189438EA.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/46152DEC-6ECB-E011-A1DB-002618943972.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/5AA78A97-6FCB-E011-A278-00304867D838.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/5E0236EB-70CB-E011-BD1E-001A928116F8.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/6619DC9A-6FCB-E011-9B02-0018F3D0967E.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/66D97CEE-6FCB-E011-AB81-003048678B00.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/687EA375-70CB-E011-BA43-0030486792DE.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/7A6D8570-6FCB-E011-9AF3-001A92810AD2.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/7A95B566-71CB-E011-8751-0026189438D8.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/7C5B3369-6FCB-E011-948D-002618943916.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/AE3E29EA-6FCB-E011-BE15-00261894396B.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/AEC978ED-6ECB-E011-990B-002618943896.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0032/1475B74E-BDCB-E011-B012-001A92810AC8.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/C474D1F0-6FCB-E011-9CA2-003048678F92.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/CEACF46C-70CB-E011-A857-002618943906.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/E47CD6EA-75CB-E011-8FC7-0018F3D096A6.root',
       '/store/relval/CMSSW_4_2_9_HLT1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V14A-v1/0031/F815BFEC-6FCB-E011-B6C7-001A92971BB2.root' 
       ]


    # FEDRawDataCollection label
    fedLabel='rawDataCollector'
else :
    # run 143657
    #dataFiles = [
    #        '/store/data/Run2010A/MinimumBias/RAW/v1/000/143/657/00FB1636-91AE-DF11-B177-001D09F248F8.root',
    #        '/store/data/Run2010A/MinimumBias/RAW/v1/000/143/657/023EB128-51AE-DF11-96D3-001D09F24682.root'                 
    #        ]
    
    # run 160413
    #dataFiles = [
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FE56EB13-BD4D-E011-B853-0030487C90D4.root',
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FCE828F2-AC4D-E011-B38A-0030487CD6B4.root',
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FC956724-CB4D-E011-8EC8-0030487CAF5E.root',
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FC8C6308-C24D-E011-85E8-0030487C7392.root',
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/F8D4B148-B34D-E011-83DF-0030487CD812.root',
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/F6F1F3D7-CB4D-E011-BAA7-0030487A195C.root'
    #        ]

    # run 165514
    dataFiles = [
                 '/store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/28C65E11-E584-E011-AED9-0030487CD700.root',
                 '/store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/44C0FC26-EE84-E011-B657-003048F1C424.root',
                 '/store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/48379944-F084-E011-8022-0030487CD178.root',
                 '/store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/4A1297CC-EC84-E011-BCF8-0030487CD6E6.root'
                 ]
    
    # FEDRawDataCollection label
    fedLabel='source'
    
    

###############
# Process setup
###############
processName = "L1GtPatternGenerator"

process = cms.Process(processName)

# Run on one orbit of events at most - more doesn't make sense
# because the pattern file can contain at most 3564 lines.
process.maxEvents = cms.untracked.PSet ( 
  input = cms.untracked.int32(3564),
)

# Input
process.source = cms.Source("PoolSource",
	  fileNames = cms.untracked.vstring(dataFiles)
	)

# Load and configure modules via global tag
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = useGlobalTag+'::All'

if useRelValSample == True :
    print 
else :
    
    import L1Trigger.Configuration.L1Trigger_custom

    # reset the prescale factors when running on data
    process = L1Trigger.Configuration.L1Trigger_custom.customiseResetPrescalesAndMasks( process )


# Global Trigger unpacker - produce decision & regional muons from readout record
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.l1GtUnpack.DaqGtInputTag = cms.InputTag(fedLabel)

# GCT unpacker - produce egamma objects / jets / etc from readout
process.load("EventFilter.GctRawToDigi.l1GctHwDigis_cfi")
process.l1GctHwDigis.inputLabel = cms.InputTag(fedLabel)

# Load and configure Pattern Generator
process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtPatternGenerator_cfi")

# Take GCT and GMT data from unpackers
gctLabel = 'l1GctHwDigis'
gmtLabel = 'l1GtUnpack'
process.l1GtPatternGenerator.GctInputTag = gctLabel
process.l1GtPatternGenerator.GmtInputTag = gmtLabel
process.l1GtPatternGenerator.CscInputTag = cms.InputTag(gmtLabel, "CSC")
process.l1GtPatternGenerator.DtInputTag = cms.InputTag(gmtLabel, "DT")
process.l1GtPatternGenerator.RpcbInputTag = cms.InputTag(gmtLabel, "RPCb")
process.l1GtPatternGenerator.RpcfInputTag = cms.InputTag(gmtLabel, "RPCf")

## enable detailed output (event no/bx per line, item tracing)
#process.l1GtPatternGenerator.DebugOutput = cms.bool(True)

# Global Trigger emulator - take input data from 
# the same sources as the pattern writer
process.load("L1Trigger.GlobalTrigger.gtDigis_cfi")
process.gtDigis.GmtInputTag = gmtLabel
process.gtDigis.GctInputTag = gctLabel
process.gtDigis.ProduceL1GtEvmRecord = False
process.gtDigis.ProduceL1GtObjectMapRecord = False
process.gtDigis.WritePsbL1GtDaqRecord = False
process.gtDigis.EmulateBxInEvent = -1

# temporary fix for L1 GT emulator configuration in hardware validation
process.gtDigis.RecordLength = cms.vint32(3, 5)
process.gtDigis.AlternativeNrBxBoardDaq = 0x101
process.gtDigis.AlternativeNrBxBoardEvm = 0x2

# explicit choice of the L1 menu, overwriting the Global Tag menu

if l1MenuSource == 'xmlFile' :
    process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff')
    process.es_prefer_l1GtParameters = cms.ESPrefer('L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml')
    
    process.l1GtTriggerMenuXml.DefXmlFile = triggerMenuXml

elif l1MenuSource == 'sqlFile' :
    if useSqlFile != '' :
        print '   Retrieve L1 trigger menu only from SQLlite file ' 
        print '       ', useSqlFile   
        print '       '

        from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
        process.l1conddb = cms.ESSource("PoolDBESSource",
                                CondDBSetup,
                                connect = cms.string('sqlite_file:' + useSqlFile),
                                toGet = cms.VPSet(cms.PSet(
                                            record = cms.string('L1GtTriggerMenuRcd'),
                                            tag = cms.string(menuDbTag))),
                                            BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
                                            )
        process.es_prefer_l1conddb = cms.ESPrefer("PoolDBESSource","l1conddb")
       
    else :
        print '   Error: no SQL file is given; please provide a valid SQL file for option sqlFile'    

else :
    print '   Printing default L1 trigger menu from Global Tag ', useGlobalTag    

# Global Trigger report (emulator)
import L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi
process.l1GtTrigReportEmul = L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi.l1GtTrigReport.clone()
process.l1GtTrigReportEmul.L1GtRecordInputTag = 'gtDigis'
# Report pass/fail for each trigger
process.l1GtTrigReportEmul.PrintVerbosity = 10

# path to be run
# - the pattern generator depends on the decision records produced by the GT emulator
process.p = cms.Path(process.l1GtUnpack * process.l1GctHwDigis * process.gtDigis * process.l1GtPatternGenerator * process.l1GtTrigReportEmul)

# switch on verbosity in L1 GT emulator
process.gtDigis.Verbosity = cms.untracked.int32(1)

# services
# uncomment / comment messages with DEBUG mode to run in DEBUG mode
process.MessageLogger = cms.Service("MessageLogger",
     destinations = cms.untracked.vstring('cout'),
     cout = cms.untracked.PSet(
         threshold = cms.untracked.string('DEBUG'), ## DEBUG mode 

         DEBUG = cms.untracked.PSet( 
             #limit = cms.untracked.int32(-1)          ## DEBUG mode, all messages  
             limit = cms.untracked.int32(0)          ## DEBUG mode, max 10 messages 
         ),
         INFO = cms.untracked.PSet(
             limit = cms.untracked.int32(-1)
         )
     ),
     debugModules = cms.untracked.vstring('1GtPatternGenerator', 'gtDigis'), ## DEBUG mode 
)

print "Setup complete."

## output - disabled by default, the pattern generator writes to a text file
#process.outputL1GtPatternGen = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('testGt_PatternGenerator_output.root')
#)

#process.outpath = cms.EndPath(process.outputL1GtPatternGen)
