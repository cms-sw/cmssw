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
    useSqlFile = '/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2011_v6/sqlFile/L1Menu_Collisions2011_v6_mc.db'
    menuDbTag = 'L1GtTriggerMenu_L1Menu_Collisions2011_v6_mc'

    
    # HI menu
    #useSqlFile = '/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_CollisionsHeavyIons2010_v2/sqlFile/L1Menu_CollisionsHeavyIons2010_v0_mc.db'
    #menuDbTag = 'L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2010_v2_mc'

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
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0068/BC61B16D-647C-E011-9972-0030486791BA.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0063/FE440F3F-847B-E011-8E8F-0018F3D096CA.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0063/E6C3406A-7A7B-E011-AAA5-002618943976.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0063/A692C906-8A7B-E011-8D69-001A92810A9E.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0063/9418B1E7-8E7B-E011-AF41-001A928116C0.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0063/6408FA2B-7E7B-E011-A3C8-001A92971B54.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0063/6271B9A6-917B-E011-AEDC-001A928116AE.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0063/3A5316CD-B37B-E011-8717-00261894383C.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0063/0E53FAD6-7E7B-E011-9514-002618943845.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0062/C0938709-727B-E011-B6EC-001BFCDBD166.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0062/BAE30EC0-767B-E011-BDE5-003048678FC6.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0062/AE2E4428-777B-E011-A75D-001A928116D6.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0062/A89B9E01-787B-E011-9A1D-0018F3D09660.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0062/9C0627FD-717B-E011-A0E2-001A92810A9E.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0062/8E6EE1A8-707B-E011-A251-003048679168.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0062/8A856689-707B-E011-ADBA-003048678DD6.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0062/88FD6678-747B-E011-9163-001A92971B84.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0062/56D4966E-727B-E011-8785-001A928116E0.root',
       '/store/relval/CMSSW_4_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V12-v2/0062/0849A316-797B-E011-80B3-0018F3D0962E.root'        
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
