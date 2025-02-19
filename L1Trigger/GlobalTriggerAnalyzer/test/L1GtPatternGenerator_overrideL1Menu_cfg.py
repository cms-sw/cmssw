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

# 
processName = "L1GtPatternGenerator"
process = cms.Process(processName)


# choose one (and only one) of the following source
# usually the global tag must be used

# data or RelVal
useRelValSample = True
#useRelValSample = False

# Frontier GlobalTag to use for EventSetup, should match source data

if useRelValSample == True :
    globalTag = 'auto:startup'
else :
    globalTag = 'auto:com10'        # GR_R_*


# L1 menu selection via L1Trigger_custom  - if True, modify correspondingly L1Trigger_custom

customL1Menu = True
#customL1Menu = False

if customL1Menu == True :
    from L1Trigger.Configuration.L1Trigger_custom import customiseL1Menu
    process=customiseL1Menu(process)

# reset all prescale factors and masks
from L1Trigger.Configuration.L1Trigger_custom import customiseResetPrescalesAndMasks
process = customiseResetPrescalesAndMasks(process)


# Input files
if useRelValSample == True :
    dataFiles = [
            '/store/relval/CMSSW_5_2_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/02B4D46B-BB51-E111-A789-003048678A76.root',
            '/store/relval/CMSSW_5_2_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/0EAB51E9-BD51-E111-8C43-003048679228.root',
            '/store/relval/CMSSW_5_2_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/16B22002-C251-E111-822B-002618FDA208.root',
            '/store/relval/CMSSW_5_2_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/28C19137-C351-E111-80CD-003048FFCBF0.root',
            '/store/relval/CMSSW_5_2_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/3034CA86-C051-E111-84F5-00304867D446.root',
            '/store/relval/CMSSW_5_2_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/30850A05-C051-E111-9BA8-002618FDA262.root',
            '/store/relval/CMSSW_5_2_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/3A0C39F6-BE51-E111-BEAE-0026189438E9.root',
            '/store/relval/CMSSW_5_2_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/7ADEB285-C151-E111-888F-0018F3D0960C.root',
            '/store/relval/CMSSW_5_2_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/A2810EE7-BB51-E111-9C4D-0026189438A0.root'
       ]

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
fedLabel='rawDataCollector'
   

###############
# Process setup
###############

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

if globalTag.count('auto') :
    from Configuration.AlCa.autoCond import autoCond
    useGlobalTag = autoCond[globalTag.replace('auto:', '')]
else :
    useGlobalTag = globalTag+'::All'    
process.GlobalTag.globaltag = useGlobalTag

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

# verbosity on for emulator
process.gtDigis.Verbosity = cms.untracked.int32(1)


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
             #limit = cms.untracked.int32(0)          ## DEBUG mode, max 10 messages 
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
