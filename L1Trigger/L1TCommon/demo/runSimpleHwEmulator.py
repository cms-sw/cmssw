import FWCore.ParameterSet.Config as cms

process = cms.Process('L1TEMULATION')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryIdeal_cff')

# Select the Message Logger output you would like to see:
process.load('FWCore.MessageService.MessageLogger_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring("file:/afs/cern.ch/work/g/ginnocen/public/skim_10_1_wd2.root")
    )

process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring('drop *',
                                           'keep *_*_*_L1TEMULATION'),
    fileName = cms.untracked.string('SimL1Emulator_Stage1_SimpleHW.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
                                           )
process.options = cms.untracked.PSet()

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag.connect = cms.string('frontier://FrontierProd/CMS_COND_31X_GLOBALTAG')
process.GlobalTag.globaltag = cms.string('POSTLS161_V12::All')

process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_PPFromRaw_cff')
process.caloStage1Digis.FirmwareVersion = cms.uint32(3)
#process.simRctDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'hcalDigis' ) )

process.p1 = cms.Path(
    process.L1TCaloStage1_PPFromRaw
    )

process.output_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.p1, process.output_step
    )

# Spit out filter efficiency at the end.
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
