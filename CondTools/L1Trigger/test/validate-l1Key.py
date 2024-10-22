import FWCore.ParameterSet.Config as cms
process = cms.Process('L1')

# Define CondDB tags
from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum
from CondTools.L1Trigger.L1O2OTags_cfi import initL1O2OTags
initL1O2OTags()

from CondTools.L1Trigger.L1CondDBSource_cff import initCondDBSource
initCondDBSource(
    process,
    inputDBConnect = 'sqlite_file:l1config.db',
    tagBaseVec = initL1O2OTags.tagBaseVec,
    includeAllTags = True
)
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(50)
)
readFiles = cms.untracked.vstring('file:Raw_160329.root')
secFiles = cms.untracked.vstring() 
process.source = cms.Source(
    'PoolSource',
    fileNames=readFiles,
    secondaryFileNames=secFiles
)

import EventFilter.EcalRawToDigi.EcalUnpackerData_cfi
process.ecalDigis = EventFilter.EcalRawToDigi.EcalUnpackerData_cfi.ecalEBunpacker.clone()
process.ecalDigis.DoRegional = False
process.ecalDigis.InputLabel = 'source'

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
process.hcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()
process.hcalDigis.InputLabel = 'source'

import EventFilter.CSCRawToDigi.cscUnpacker_cfi
process.muonCSCDigis = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone()
process.muonCSCDigis.InputObjects = 'source'

import EventFilter.DTRawToDigi.dtunpacker_cfi
process.muonDTDigis = EventFilter.DTRawToDigi.dtunpacker_cfi.muonDTDigis.clone()
process.muonDTDigis.inputLabel = 'source'

import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
process.muonRPCDigis = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone()
process.muonRPCDigis.InputLabel = 'source'

# run trigger primitive generation on unpacked digis, then central L1
process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
process.load('L1Trigger.Configuration.SimL1Emulator_cff')

# set the new input tags after RawToDigi for the TPG producers
process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
    cms.InputTag('hcalDigis'), 
    cms.InputTag('hcalDigis')
)
process.simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'
process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag(
    'muonCSCDigis',
    'MuonCSCComparatorDigi'
)
process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag(
    'muonCSCDigis',
    'MuonCSCWireDigi'
)
process.simRpcTriggerDigis.label = 'muonRPCDigis'
process.simRpcTechTrigDigis.RPCDigiLabel = 'muonRPCDigis'

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
#process.GlobalTag.globaltag = 'GR10_H_V4::All'
#process.GlobalTag.globaltag = 'GR10_H_V8::All'
process.GlobalTag.globaltag = 'GR_H_V20::All'

process.GlobalTag.toGet = cms.VPSet()
process.GlobalTag.toGet.append(
    cms.PSet(
        record = cms.string('DTCCBConfigRcd'),
        tag = cms.string('DT_config_V04'),
    ) )
process.GlobalTag.toGet.append(
    cms.PSet(
        record = cms.string('DTKeyedConfigListRcd'),
        tag = cms.string('DT_keyedConfListIOV_V01'),
    ) )
process.GlobalTag.toGet.append(
    cms.PSet(
        record = cms.string('DTKeyedConfigContainerRcd'),
        tag = cms.string('DT_keyedConfBricks_V01'),
    ) )

process.p = cms.Path(
    process.ecalDigis * process.hcalDigis 
    * process.CaloTriggerPrimitives 
    * process.muonDTDigis 
    * process.muonCSCDigis 
    * process.muonRPCDigis 
    * process.SimL1Emulator
)

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['*']

process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
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
#process.output = cms.OutputModule("PoolOutputModule",
#    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
#    fileName = cms.untracked.string('L1EmulatorFromRaw.root'),
#    dataset = cms.untracked.PSet(
#        dataTier = cms.untracked.string("\'GEN-SIM-DIGI-RAW-HLTDEBUG\'"),
#        filterName = cms.untracked.string('')
#    )
#)
#process.out_step = cms.EndPath(process.output)
