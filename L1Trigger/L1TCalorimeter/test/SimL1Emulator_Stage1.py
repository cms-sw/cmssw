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
    input = cms.untracked.int32(100)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring("root://xrootd.unl.edu//store/mc/Fall13dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/tsg_PU40bx25_POSTLS162_V2-v1/00005/02B79593-F47F-E311-8FF6-003048FFD796.root")
    )


process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
#    outputCommands = cms.untracked.vstring('keep *',
#                                           'drop FEDRawDataCollection_rawDataRepacker_*_*',
#                                           'drop FEDRawDataCollection_virginRawDataRepacker_*_*'),
    outputCommands = cms.untracked.vstring('drop *',
                                           'keep *_*_*_L1TEMULATION'),
    fileName = cms.untracked.string('SimL1Emulator_Stage1_PP.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
                                           )
process.options = cms.untracked.PSet()

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag.connect = cms.string('frontier://FrontierProd/CMS_COND_31X_GLOBALTAG')
process.GlobalTag.globaltag = cms.string('POSTLS162_V2::All')

process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_PPFromRaw_cff')

## changes to L1 algorithms begin here, the list is exhaustive.
## commented values should be the default
## see L1Trigger/L1TCalorimeter/python/l1tCaloStage1Digis_cfi.py for more info
#process.caloStage1Digis.egRelativeJetIsolationCut = cms.double(0.5)
#process.caloStage1Digis.tauRelativeJetIsolationCut = cms.double(1.)
#process.caloStage1Digis.regionETCutForHT = cms.uint32(7)
#process.caloStage1Digis.regionETCutForMET = cms.uint32(0)
#process.caloStage1Digis.minGctEtaForSums = cms.int32(4)
#process.caloStage1Digis.maxGctEtaForSums = cms.int32(17)

#process.l1tCaloParams.egLsb = cms.double(1.0),
#process.l1tCaloParams.egSeedThreshold = cms.double(1.),
#process.l1tCaloParams.jetLsb = cms.double(0.5),
#process.l1tCaloParams.jetSeedThreshold = cms.double(10.),
#process.l1tCaloParams.etSumLsb = cms.double(0.5),
#process.l1tCaloParams.etSumEtaMin = cms.vint32(-999, -999, -999, -999),
#process.l1tCaloParams.etSumEtaMax = cms.vint32(999,  999,  999,  999),
#process.l1tCaloParams.etSumEtThreshold = cms.vdouble(0.,  0.,   0.,   0.)

process.p1 = cms.Path(
    process.L1TCaloStage1_PPFromRaw
    )

process.output_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.p1, process.output_step
    )

# Spit out filter efficiency at the end.
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
