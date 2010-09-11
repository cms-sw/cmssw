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
    useGlobalTag = "START38_V10"
else :
    useGlobalTag = "GR10_P_V9"
    

#l1MenuSource='globalTag'
l1MenuSource='sqlFile'
#l1MenuSource='xmlFile'


if l1MenuSource == 'sqlFile' :
    # the menu will be read from the SQL file instead of the global tag
    useSqlFile = '/afs/cern.ch/user/g/ghete/public/L1Menu/sqlFile/L1Menu_Commissioning2010_v4_mc.db'
    menuDbTag = 'L1GtTriggerMenu_L1Menu_Commissioning2010_v4_mc'
    #useSqlFile = '/afs/cern.ch/user/g/ghete/public/L1Menu/sqlFile/L1Menu_MC2010_v0_mc.db'
    #menuDbTag = 'L1GtTriggerMenu_L1Menu_MC2010_v0_mc'
elif l1MenuSource == 'xmlFile' :
    # explicit choice of the L1 menu
    # un-comment the corresponding menu in the list of the menus
    triggerMenuXml = 'L1Menu_Commissioning2010_v3.xml'
    print '   Using', triggerMenuXml, ' file to overwrite the L1 menu from the global tag'    

else :
    print '   Using default L1 trigger menu from Global Tag ', useGlobalTag    

# Input files
if useRelValSample == True :
    dataFiles = [
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/DAA27EF5-5069-DF11-9B53-002618943982.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/D23295F2-5069-DF11-8EAD-002354EF3BDE.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/BEEC3A7D-5269-DF11-B4BD-003048678A88.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/BCBC54C8-6069-DF11-A680-003048678ADA.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/B28BFFF5-4F69-DF11-8BA5-002618943969.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/A2CF55FD-4F69-DF11-AD67-00261894395F.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/8AB3AA68-5069-DF11-9629-001A928116B8.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/8813F4FB-5169-DF11-A602-0026189438DB.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/72382A69-4F69-DF11-86D5-0018F3D096D2.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/3C83A8F2-5069-DF11-8691-002618943856.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/36D79AE9-5069-DF11-BEC3-00261894397D.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/101CA9FE-4F69-DF11-BA81-002618943979.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/0CB86B26-5A69-DF11-8D46-00261894380A.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/064A62C7-5E69-DF11-9C52-003048678F84.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/F6E38566-4E69-DF11-8E5E-0030486790BE.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/F4CBE7D7-4E69-DF11-92FB-0026189438E3.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/C2D99BDA-4D69-DF11-A886-002618943934.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/BCD64BD9-4D69-DF11-B135-00261894396B.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/5C5DF74D-4D69-DF11-B0BA-00261894387E.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/200B80D9-4E69-DF11-BBE9-00261894396B.root'              
             ]
    # FEDRawDataCollection label
    fedLabel='rawDataCollector'
else :
    # run 143657
    dataFiles = [
            '/store/data/Run2010A/MinimumBias/RAW/v1/000/143/657/00FB1636-91AE-DF11-B177-001D09F248F8.root',
            '/store/data/Run2010A/MinimumBias/RAW/v1/000/143/657/023EB128-51AE-DF11-96D3-001D09F24682.root'                 
             ]
    # FEDRawDataCollection label
    fedLabel='source'


###############
# Process setup
###############
processName = "L1GtPatternGenerator"

process = cms.Process(processName)

# Run on one arbit of events at most - more doesn't make sense
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
process.l1GtTrigReportEmul.PrintVerbosity = 2

# path to be run
# - the pattern generator depends on the decision records produced by the GT emulator
process.p = cms.Path(process.l1GtUnpack * process.l1GctHwDigis * process.gtDigis * process.l1GtPatternGenerator * process.l1GtTrigReportEmul)

# services
# uncomment / comment messages with DEBUG mode to run in DEBUG mode
process.MessageLogger = cms.Service("MessageLogger",
     destinations = cms.untracked.vstring('cout'),
     cout = cms.untracked.PSet(
         #threshold = cms.untracked.string('DEBUG'), ## DEBUG mode 

         DEBUG = cms.untracked.PSet( 
             #limit = cms.untracked.int32(-1)          ## DEBUG mode, all messages  
             #limit = cms.untracked.int32(10)         ## DEBUG mode, max 10 messages 
         ),
         INFO = cms.untracked.PSet(
             limit = cms.untracked.int32(-1)
         )
     ),
     #debugModules = cms.untracked.vstring('1GtPatternGenerator', 'gtDigis'), ## DEBUG mode 
)

print "Setup complete."

## output - disabled by default, the pattern generator writes to a text file
#process.outputL1GtPatternGen = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('testGt_PatternGenerator_output.root')
#)

#process.outpath = cms.EndPath(process.outputL1GtPatternGen)
