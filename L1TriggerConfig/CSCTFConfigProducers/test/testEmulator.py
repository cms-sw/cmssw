# from https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1O2OOperations
# to test the emulator from the .cms cluster retrieving the global tag
#
# The reference is L1Trigger/Configuration/test/L1EmulatorFromRaw_cfg.py
#
# test.py starts here
# 
#
import FWCore.ParameterSet.Config as cms
process = cms.Process('L1')

##from CondTools.L1Trigger.L1CondDBSource_cff import initCondDBSource
##initCondDBSource(
##    process,
##    inputDBConnect = 'sqlite_file:l1config.db',
##    tagBase = 'CRAFT09_hlt',
##    includeAllTags = True
##)
## ss.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
## ss.GlobalTag.globaltag = 'CRFT9_37R_V0::All'
## ss.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_31X_GLOBALTAG"
##  on srv-c2d05-12

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(50)
)
readFiles = cms.untracked.vstring('file:Raw.root')
secFiles = cms.untracked.vstring() 
process.source = cms.Source(
    'PoolSource',
    fileNames=readFiles,
    secondaryFileNames=secFiles
)

## import EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi
## process.ecalDigis = EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi.ecalEBunpacker.clone()
## process.ecalDigis.DoRegional = False
## process.ecalDigis.InputLabel = 'source'
## 
## import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
## process.hcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()
## process.hcalDigis.InputLabel = 'source'
## 
## import EventFilter.CSCRawToDigi.cscUnpacker_cfi
## process.muonCSCDigis = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone()
## process.muonCSCDigis.InputObjects = 'source'
## 
## import EventFilter.DTRawToDigi.dtunpacker_cfi
## process.muonDTDigis = EventFilter.DTRawToDigi.dtunpacker_cfi.muonDTDigis.clone()
## process.muonDTDigis.inputLabel = 'source'
## 
## import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
## process.muonRPCDigis = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone()
## process.muonRPCDigis.InputLabel = 'source'
## 
## # run trigger primitive generation on unpacked digis, then central L1
## process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
## process.load('L1Trigger.Configuration.SimL1Emulator_cff')
## 
## # set the new input tags after RawToDigi for the TPG producers
## process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
## process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
##     cms.InputTag('hcalDigis'), 
##     cms.InputTag('hcalDigis')
## )
## process.simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'
## process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag(
##     'muonCSCDigis',
##     'MuonCSCComparatorDigi'
## )
## process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag(
##     'muonCSCDigis',
##     'MuonCSCWireDigi'
## )
## process.simRpcTriggerDigis.label = 'muonRPCDigis'
## process.simRpcTechTrigDigis.RPCDigiLabel = 'muonRPCDigis'

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.GlobalTag.globaltag = 'GR10_H_V4::All'

process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_31X_GLOBALTAG"

## process.p = cms.Path(
##     process.ecalDigis * process.hcalDigis 
##     * process.CaloTriggerPrimitives 
##     * process.muonDTDigis 
##     * process.muonCSCDigis 
##     * process.muonRPCDigis 
##     * process.SimL1Emulator
## )

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['*']
process.MessageLogger.categories = ['*']
process.MessageLogger.destinations = ['cout']
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    DEBUG=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    INFO=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    WARNING=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    ERROR=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    )
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    fileName = cms.untracked.string('L1EmulatorFromRaw.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string("\'GEN-SIM-DIGI-RAW-HLTDEBUG\'"),
        filterName = cms.untracked.string('')
    )
)
process.out_step = cms.EndPath(process.output)
#
# test.py ends here
#
# CSC Track Finder emulator (copy-paste from L1Trigger/Configuration/python/SimL1Emulator_cff.py + little modifications)
# Little pieces of configuration, taken here and there
#process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("L1TriggerConfig.CSCTFConfigProducers.L1CSCTFConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")


# CSC TF (copy-paste L1Trigger/Configuration/python/L1RawToDigi_cff.py + little modifications)
import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
process.csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
process.csctfDigis.producer = 'source'
#

import L1Trigger.CSCTrackFinder.csctfDigis_cfi

# ------------------------------------------------------------------------------------------------
# IMPORTANT:
# ---------
#
# IF YOU WANT TO CONFIGURE THE EMULATOR VIA EventSetup (O2O mechanism or fake producer) the
# option initializeFromPSet in L1Trigger/CSCTrackFinder/python/csctfTrackDigis_cfi.py
# has to be set to False: initializeFromPSet = cms.bool(False)
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
process.simCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("csctfDigis")
process.simCsctfTrackDigis.SectorProcessor.initializeFromPSet = cms.bool(False)
process.simCsctfTrackDigis.useDT = cms.bool(False)

# ------------------------------------------------------------------------------------------------
## # Following important parameters have to be set for singles by hand
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME1a = cms.bool(True)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME1b = cms.bool(True)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME2  = cms.bool(True)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME3  = cms.bool(True)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME4  = cms.bool(True)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_MB1a = cms.bool(False)
## process.simCsctfTrackDigis.SectorProcessor.trigger_on_MB1d = cms.bool(False)
## process.simCsctfTrackDigis.SectorProcessor.singlesTrackPt     = cms.uint32(255)
## process.simCsctfTrackDigis.SectorProcessor.singlesTrackOutput = cms.uint32(1)

process.p = cms.Path(process.csctfDigis*process.simCsctfTrackDigis)

