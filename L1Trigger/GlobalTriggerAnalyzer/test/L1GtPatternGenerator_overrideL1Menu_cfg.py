from __future__ import print_function
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
useRelValSample = False

# Frontier GlobalTag to use for EventSetup, should match source data

if useRelValSample == True :
    globalTag = 'auto:startup'
else :
    globalTag = 'auto:com10'        # GR_R_*


# L1 menu selection via L1Trigger_custom  - if True, modify correspondingly L1Trigger_custom

customL1Menu = True
customL1Menu = False

if customL1Menu == True :
    from L1Trigger.Configuration.L1Trigger_custom import customiseL1Menu
    process=customiseL1Menu(process)

# reset all prescale factors and masks
from L1Trigger.Configuration.L1Trigger_custom import customiseResetPrescalesAndMasks
process = customiseResetPrescalesAndMasks(process)


# Input files

# initialize list of selected events and luminosity segments
selectedEvents = cms.untracked.VEventRange()
selectedLumis= cms.untracked.VLuminosityBlockRange()

if useRelValSample == True :
    
    dataFiles = [
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/0ABED808-FE79-E111-A8CB-003048FFCBFC.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/1ED8ACDC-FF79-E111-BEB8-0018F3D096C6.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/38A7298B-FF79-E111-8E7D-003048FFD736.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/48D6CA06-FE79-E111-8A08-001A92811716.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/521004DD-FF79-E111-860D-003048679150.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/52389B0B-FE79-E111-81E4-003048FFCBA4.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/54738A02-FE79-E111-B495-0030486790B0.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/5A921406-007A-E111-BFE5-003048FFD736.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/5E86D805-FE79-E111-9EC4-003048FFD7A2.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/66FCF29D-FF79-E111-9DCE-0018F3D096C6.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/6C7BB304-FE79-E111-8FAC-001A92810AEE.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/784AC1AB-FF79-E111-9C08-003048679150.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/929ACBC3-FF79-E111-805A-003048FFD736.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/A0D7350B-FE79-E111-8E0D-003048FFCC2C.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/C268AF61-FD79-E111-AB3D-00261894397D.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/D237D303-FE79-E111-B38B-001BFCDBD100.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/D428BF06-FE79-E111-8445-003048678FEA.root',
       '/store/relval/CMSSW_5_2_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V5-v1/0043/EEB8C84A-FC79-E111-9BBB-002618943985.root' ]


else :
    # run 143657
    
    #dataType = 'RAW'
    #dataFiles = [
    #        '/store/data/Run2010A/MinimumBias/RAW/v1/000/143/657/00FB1636-91AE-DF11-B177-001D09F248F8.root',
    #        '/store/data/Run2010A/MinimumBias/RAW/v1/000/143/657/023EB128-51AE-DF11-96D3-001D09F24682.root'                 
    #        ]
    
    # run 160413
    #dataType = 'RAW'
    #dataFiles = [
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FE56EB13-BD4D-E011-B853-0030487C90D4.root',
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FCE828F2-AC4D-E011-B38A-0030487CD6B4.root',
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FC956724-CB4D-E011-8EC8-0030487CAF5E.root',
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/FC8C6308-C24D-E011-85E8-0030487C7392.root',
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/F8D4B148-B34D-E011-83DF-0030487CD812.root',
    #        '/store/data/Run2011A/MinimumBias/RAW/v1/000/160/413/F6F1F3D7-CB4D-E011-BAA7-0030487A195C.root'
    #        ]

#    # run 165514
#    dataType = 'RAW'
#    dataFiles = [
#                 '/store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/28C65E11-E584-E011-AED9-0030487CD700.root',
#                 '/store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/44C0FC26-EE84-E011-B657-003048F1C424.root',
#                 '/store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/48379944-F084-E011-8022-0030487CD178.root',
#                 '/store/data/Run2011A/MinimumBias/RAW/v1/000/165/514/4A1297CC-EC84-E011-BCF8-0030487CD6E6.root'
#                 ]
    
    # sample '205666.A.storageManager'
    dataType = 'StreamFile'
    dataFiles = [
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0001.A.storageManager.00.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0021.A.storageManager.01.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0041.A.storageManager.02.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0061.A.storageManager.03.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0081.A.storageManager.04.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0101.A.storageManager.05.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0121.A.storageManager.06.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0141.A.storageManager.07.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0161.A.storageManager.08.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0181.A.storageManager.09.0000.dat',                   
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0201.A.storageManager.10.0000.dat',                   
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0221.A.storageManager.11.0000.dat',                   
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0241.A.storageManager.12.0000.dat',                   
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0261.A.storageManager.13.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0281.A.storageManager.14.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0301.A.storageManager.15.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0321.A.storageManager.00.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0341.A.storageManager.01.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0361.A.storageManager.02.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0381.A.storageManager.03.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0401.A.storageManager.04.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0421.A.storageManager.05.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0441.A.storageManager.06.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0461.A.storageManager.07.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0481.A.storageManager.08.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0501.A.storageManager.09.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0521.A.storageManager.10.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0541.A.storageManager.11.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0561.A.storageManager.12.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0581.A.storageManager.13.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0601.A.storageManager.14.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0621.A.storageManager.15.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0641.A.storageManager.00.0000.dat',
#            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0661.A.storageManager.01.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0681.A.storageManager.02.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0701.A.storageManager.03.0000.dat',
            'file:/afs/cern.ch/user/g/ghete/work/L1Trigger/data/Data.00205666.0721.A.storageManager.04.0000.dat'                     
                 ]
 
    selectedEvents = cms.untracked.VEventRange(
                                    '205666:101:125069341',
                                    '205666:161:238160322',
                                    '205666:281:443371795',
                                    '205666:341:537654884',
                                    '205666:361:567121052', 
                                    '205666:361:567121053-205666:721:MAX'
                                    )
            


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

# input source according to data type
if dataType == 'StreamFile' :
    process.source = cms.Source("NewEventStreamFileReader", 
                                fileNames=cms.untracked.vstring(dataFiles),
                                lumisToProcess = selectedLumis,
                                eventsToProcess = selectedEvents
                                )
else :        
    process.source = cms.Source ('PoolSource', 
                                 fileNames=cms.untracked.vstring(dataFiles), 
                                 lumisToProcess = selectedLumis,
                                 eventsToProcess = selectedEvents
                                 )


# Load and configure modules via global tag
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

if globalTag.count('auto') :
    from Configuration.AlCa.autoCond import autoCond
    useGlobalTag = autoCond[globalTag.replace('auto:', '')]
else :
    useGlobalTag = globalTag    
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

# switch on verbosity in L1 GT emulator
#process.gtDigis.Verbosity = cms.untracked.int32(1)



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

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtPatternGenerator', 'gtDigis']

process.MessageLogger.cerr.enable = False
process.MessageLogger.cerr.default.limit = 0
process.MessageLogger.cerr.FwkReport.limit = 0
process.MessageLogger.cerr.FwkSummary.limit = 0

process.MessageLogger.files.L1GtPatternGenerator_debug = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtPatternGenerator = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(0) ) 
        )

process.MessageLogger.files.L1GtPatternGenerator_info = cms.untracked.PSet( 
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtPatternGenerator = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.files.L1GtPatternGenerator_warning = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtPatternGenerator = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(0) ) 
        )

process.MessageLogger.files.L1GtPatternGenerator_error = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtPatternGenerator = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
       )

print("Setup complete.")

## output - disabled by default, the pattern generator writes to a text file
#process.outputL1GtPatternGen = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('testGt_PatternGenerator_output.root')
#)

#process.outpath = cms.EndPath(process.outputL1GtPatternGen)
