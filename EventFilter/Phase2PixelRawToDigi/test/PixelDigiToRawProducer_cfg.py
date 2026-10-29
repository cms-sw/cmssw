# Phase-2 IT packer/unpacker round-trip:
#   pdigi -> per-chip bitstream -> raw -> pdigi (RawToPixelProducer unpacks in one go)

import os
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

_src = os.path.join(os.environ['CMSSW_BASE'], 'src')
_db = os.path.join(_src, 'EventFilter/Phase2PixelRawToDigi/test/OTandITDTCCablingMap.db')

options = VarParsing.VarParsing('analysis')
options.register('outputEDM', 'output_file.root',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'EDM output file')
options.register('cablingDB', 'sqlite_file:' + _db,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'Sqlite cabling-map DB connect string')
options.setDefault('maxEvents', 20)
options.parseArguments()

files = list(options.inputFiles) or [
    'root://cms-xrd-global.cern.ch//store/relval/CMSSW_20_1_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/'
    'PU_150X_mcRun4_realistic_v1_SpecialRV296_Run4D112-v1/2830000/5f358aba-be26-403b-a2ce-17c07b6b0717.root'
]

process = cms.Process('USER', Phase2C17I13M9)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D112Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))
process.source = cms.Source('PoolSource', fileNames=cms.untracked.vstring(files))

process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = options.cablingDB
process.PoolDBESSource = cms.ESSource('PoolDBESSource', process.CondDB,
    toGet=cms.VPSet(cms.PSet(
        record=cms.string('TrackerDetToDTCELinkCablingMapRcd'),
        tag=cms.string('DTCCablingMapProducerUserRun'),
    )),
)
process.es_prefer_local_cabling = cms.ESPrefer('PoolDBESSource', '')

process.PixelToBitStreamProducer = cms.EDProducer('PixelToBitStreamProducer',
    src=cms.InputTag('generalTracks'),
    siPixelDigi=cms.InputTag('simSiPixelDigis', 'Pixel'),
)
process.BitStreamToRawProducer = cms.EDProducer('BitStreamToRawProducer',
    Phase2ITChipBitStream=cms.InputTag('PixelToBitStreamProducer'),
)
process.RawToPixelProducer = cms.EDProducer('RawToPixelProducer',
    fedRawDataCollection=cms.InputTag('BitStreamToRawProducer'),
)

process.out = cms.OutputModule('PoolOutputModule',
    fileName=cms.untracked.string(options.outputEDM),
    outputCommands=cms.untracked.vstring(
        'drop *',
        'keep RawDataBuffer_*_*_*',
        'keep *_RawToPixelProducer_*_*',  # unpacked digis
    ),
)

process.p = cms.Path(process.PixelToBitStreamProducer * process.BitStreamToRawProducer * process.RawToPixelProducer)
process.e = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.p, process.e)
process.options.numberOfThreads = cms.untracked.uint32(1)
