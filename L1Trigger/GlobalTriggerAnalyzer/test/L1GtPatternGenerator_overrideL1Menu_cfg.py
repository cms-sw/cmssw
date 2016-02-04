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
    #
    # pp menu
    #useSqlFile = '/afs/cern.ch/user/g/ghete/public/L1Menu/sqlFile/L1Menu_Collisions2010_v0_mc.db'
    #menuDbTag = 'L1GtTriggerMenu_L1Menu_Collisions2010_v0_mc'
    #
    # HI menu
    useSqlFile = '/afs/cern.ch/user/g/ghete/public/L1Menu/sqlFile/L1Menu_CollisionsHeavyIons2010_v0_mc.db'
    menuDbTag = 'L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2010_v0_mc'
    #
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
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/FEC54C30-612B-E011-9836-00261894386E.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/FC3DBA1B-652B-E011-82F4-00261894392B.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/FA15AF21-622B-E011-B577-0018F3D096F8.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/F4F2BFA3-612B-E011-8306-002618943925.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/F4A40216-642B-E011-A110-002618943877.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/EED30BA6-602B-E011-88DB-002618943918.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/EE0C58A1-622B-E011-B3D6-001A928116B2.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/E6EF1D13-6E2B-E011-B7DF-002618943978.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/E0ADF921-622B-E011-8D4F-00261894384A.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/DE9FEF8F-772B-E011-A1AE-002618FDA279.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/DC8AC8A3-602B-E011-B899-0026189438E2.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/DC3C8826-622B-E011-9364-0026189438F8.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/DACBEEA6-612B-E011-B52D-002618943858.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/C8A81FCE-662B-E011-9823-0026189438DE.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/C800D4A8-612B-E011-9456-00261894387B.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/B6B6712B-612B-E011-96FE-002354EF3BDC.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/B432369E-5F2B-E011-A4E4-002354EF3BD2.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/9E3FE716-712B-E011-9E34-0018F3D096E6.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/9C65C19D-5F2B-E011-ADBE-00261894380D.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/9AC31F1D-622B-E011-A3F3-002618943832.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/9675C79C-5F2B-E011-97C9-002618FDA204.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/92489B9C-622B-E011-9A1D-0026189438E7.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/8A51CA2C-612B-E011-87F3-0026189438AB.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/84EEA290-6F2B-E011-A16E-001A92810A9E.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_highstats-v1/0004/82333298-632B-E011-955C-002618943898.root'
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
    
    # reset the prescale factors when running on data
    process.load("L1TriggerConfig.L1GtConfigProducers.L1GtPrescaleFactorsAlgoTrigConfig_cff")
    process.es_prefer_l1GtTriggerMaskAlgoTrig = cms.ESPrefer(
                    "L1GtPrescaleFactorsAlgoTrigTrivialProducer","l1GtPrescaleFactorsAlgoTrig")

    process.load("L1TriggerConfig.L1GtConfigProducers.L1GtPrescaleFactorsTechTrigConfig_cff")
    process.es_prefer_l1GtTriggerMaskAlgoTrig = cms.ESPrefer(
                    "L1GtPrescaleFactorsTechTrigTrivialProducer","l1GtPrescaleFactorsTechTrig")
    

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
process.l1GtTrigReportEmul.PrintVerbosity = 10

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
