# Phase-2 IT pipeline: pdigi -> per-chip bitstream -> Aurora formatting

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process('USER', Phase2C17I13M9)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D102Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(20))

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/relval/CMSSW_14_1_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_140X_mcRun4_realistic_v3_SpecialRV296_Run4D112-v1/2590000/b1caa2db-9810-4d64-bc7c-f02d1dbe652d.root")
)

# Local DTC cabling map (sqlite). Required by PixelToBitStreamProducer.
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = 'sqlite_file:OTandITDTCCablingMap.db'
process.PoolDBESSource = cms.ESSource('PoolDBESSource', process.CondDB,
    DumpStat = cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerDetToDTCELinkCablingMapRcd'),
        tag    = cms.string('DTCCablingMapProducerUserRun'),
    )),
)
process.es_prefer_local_cabling = cms.ESPrefer('PoolDBESSource', '')

process.PixelToBitStreamProducer = cms.EDProducer('PixelToBitStreamProducer',
    src         = cms.InputTag('generalTracks'),
    siPixelDigi = cms.InputTag('simSiPixelDigis', 'Pixel'),
)
process.BitStreamToAuroraProducer = cms.EDProducer('BitStreamToAuroraProducer',
    Phase2ITChipBitStream = cms.InputTag('PixelToBitStreamProducer'),
    eventsPerStream       = cms.uint32(16),    # NE unaligned mode streaming
    serviceBlockInterval  = cms.uint32(50),    # ND aurora block setting
)

# Keep only events that close a complete NE-stream group.
process.auroraFilter = cms.EDFilter('BooleanFlagFilter',
    inputLabel      = cms.InputTag('BitStreamToAuroraProducer', 'isComplete'),
    reverseDecision = cms.bool(False),
)

process.auroraOutput = cms.OutputModule('PoolOutputModule',
    fileName       = cms.untracked.string('aurora_output.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_BitStreamToAuroraProducer_*_*',
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('user_step')),
)

process.user_step = cms.Path(
    process.PixelToBitStreamProducer
    * process.BitStreamToAuroraProducer
    * process.auroraFilter
)
process.endjob_step    = cms.EndPath(process.endOfProcess)
process.output_endpath = cms.EndPath(process.auroraOutput)

process.schedule = cms.Schedule(
    process.user_step,
    process.endjob_step,
    process.output_endpath,
)
