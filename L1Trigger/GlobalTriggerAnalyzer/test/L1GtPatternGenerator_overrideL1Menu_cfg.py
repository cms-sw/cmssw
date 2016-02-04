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
    useGlobalTag = "START311_V2"
else :
    useGlobalTag = "GR10_P_V12"
    

#l1MenuSource='globalTag'
l1MenuSource='sqlFile'
#l1MenuSource='xmlFile'


if l1MenuSource == 'sqlFile' :
    # the menu will be read from the SQL file instead of the global tag
    
    
    # pp menu
    useSqlFile = '/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2011_v1/sqlFile/L1Menu_Collisions2011_v1_mc.db'
    menuDbTag = 'L1GtTriggerMenu_L1Menu_Collisions2011_v1_mc'

    
    # HI menu
    #useSqlFile = '/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_CollisionsHeavyIons2010_v2/sqlFile/L1Menu_CollisionsHeavyIons2010_v0_mc.db'
    #menuDbTag = 'L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2010_v2_mc'

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
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/D60EB71B-0345-E011-A1D2-00261894387A.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/D076E11B-0345-E011-A6FB-0026189438B4.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/D0402114-0145-E011-9D3D-002618943930.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/C657678E-0145-E011-B585-0026189438C1.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/BC2EBBEE-2045-E011-B77F-003048678FC6.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/AE716016-0B45-E011-97F2-00304867906C.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/92D6488F-0145-E011-A3CF-0030486792B6.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/825D0C96-0145-E011-BA93-0018F3D09652.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/80759312-0145-E011-960B-002618B27F8A.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/78D79312-0145-E011-9F14-003048679008.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/4ECB0115-0845-E011-B61C-0018F3D0965C.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/4C156416-0245-E011-AF26-002618943879.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/4498FE1E-0245-E011-8455-002354EF3BDA.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/24588390-0245-E011-8B11-0018F3D096B6.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/24172521-0245-E011-8B97-002618943982.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/1C82EC1D-0D45-E011-8962-001A92810AD2.root',
       '/store/relval/CMSSW_4_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0019/0416F308-1345-E011-B08E-0026189438F9.root' 
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
    dataFiles = [
            '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FE56EB13-BD4D-E011-B853-0030487C90D4.root',
            '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FCE828F2-AC4D-E011-B38A-0030487CD6B4.root',
            '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FC956724-CB4D-E011-8EC8-0030487CAF5E.root',
            '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FC8C6308-C24D-E011-85E8-0030487C7392.root',
            '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/F8D4B148-B34D-E011-83DF-0030487CD812.root',
            '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/F6F1F3D7-CB4D-E011-BAA7-0030487A195C.root'
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
             limit = cms.untracked.int32(-1)          ## DEBUG mode, all messages  
             #limit = cms.untracked.int32(10)          ## DEBUG mode, max 10 messages 
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
