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
    input = cms.untracked.int32(10)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    #fileNames = cms.untracked.vstring("file:22610530-FC24-E311-AF35-003048FFD7C2.root")
    #fileNames = cms.untracked.vstring("file:/mnt/hadoop/cms/store/user/icali/HIHighPt/HIHIHighPt_RAW_Skim_HLT_HIFullTrack14/4d786c9deacb28bba8fe5ed87e99b9e4/SD_HIFullTrack14_975_1_SZU.root")
    fileNames = cms.untracked.vstring("root://xrootd.cmsaf.mit.edu//store/user/icali/HIMinBiasUPC/HIMinBiasUPC_Skim_HLT_HIMinBiasHfOrBSC_v2/35880fcf9fb9fd84b27cd1405e09ffd1/SD_MinBiasHI_977_1_tba.root")
    #fileNames = cms.untracked.vstring("/store/relval/CMSSW_7_1_0_pre8/RelValHydjetQ_MinBias_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_SHI71_V7-v1/00000/00AAA72D-C1E3-E311-AE93-02163E00F43C.root")
    )


process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
#    outputCommands = cms.untracked.vstring('keep *',
#                                           'drop FEDRawDataCollection_rawDataRepacker_*_*',
#                                           'drop FEDRawDataCollection_virginRawDataRepacker_*_*'),
    outputCommands = cms.untracked.vstring('drop *',
                                           'keep *_*_*_L1TEMULATION',
                                           'drop *_ecalDigis_*_*',
                                           'drop *_hcalDigis_*_*'),
    fileName = cms.untracked.string('SimL1Emulator_Stage1_HI.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
                                           )
process.options = cms.untracked.PSet()

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag.connect = cms.string('frontier://FrontierProd/CMS_COND_31X_GLOBALTAG')
#process.GlobalTag.globaltag = cms.string('POSTLS162_V2::All')
#for HI Data
process.GlobalTag = GlobalTag(process.GlobalTag, 'GR_P_V27A::All', '')

process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_HIFromRaw_cff')

## changes to L1 algorithms begin here, the list is exhaustive.
## commented values should be the default
## see L1Trigger/L1TCalorimeter/python/l1tCaloStage1Digis_cfi.py for more info
#process.caloStage1Digis.FirmwareVersion = cms.uint32(3) # 1=HI algos, 2=PP algos

#process.l1tCaloStage1Params.egLsb = cms.double(1.0),
#process.l1tCaloStage1Params.egSeedThreshold = cms.double(1.),
#process.l1tCaloStage1Params.jetLsb = cms.double(0.5),
#process.l1tCaloStage1Params.jetSeedThreshold = cms.double(0.), #HI doesn't need a jet seed threshold to reduce rate
#process.l1tCaloStage1Params.etSumLsb = cms.double(0.5),
#process.l1tCaloStage1Params.etSumEtaMin = cms.vint32(-999, -999, -999, -999),
#process.l1tCaloStage1Params.etSumEtaMax = cms.vint32(999,  999,  999,  999),
#process.l1tCaloStage1Params.etSumEtThreshold = cms.vdouble(0.,  0.,   0.,   0.)

process.p1 = cms.Path(
    process.L1TCaloStage1_HIFromRaw
    )

process.output_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.p1, process.output_step
    )

# Spit out filter efficiency at the end.
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
