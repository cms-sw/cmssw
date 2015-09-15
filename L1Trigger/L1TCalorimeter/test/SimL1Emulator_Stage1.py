import FWCore.ParameterSet.Config as cms

process = cms.Process('L1TEMULATION')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
## process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryIdeal_cff')

# Select the Message Logger output you would like to see:
process.load('FWCore.MessageService.MessageLogger_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-200)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring("/store/relval/CMSSW_7_5_0_pre4/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_75_V1-v1/00000/0CD12657-DAF7-E411-91F2-002618943910.root")
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
## from Configuration.AlCa.GlobalTag import GlobalTag
## process.GlobalTag.connect = cms.string('frontier://FrontierProd/CMS_COND_31X_GLOBALTAG')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag.globaltag = cms.string('MCRUN2_75_V1')

process.load('L1Trigger.L1TCalorimeter.caloConfigStage1PP_cfi')

process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_PPFromRaw_cff')

# GT
from L1Trigger.Configuration.SimL1Emulator_cff import simGtDigis
process.simGtDigis = simGtDigis.clone()
process.simGtDigis.GmtInputTag = 'simGmtDigis'
process.simGtDigis.GctInputTag = 'simCaloStage1LegacyFormatDigis'
process.simGtDigis.TechnicalTriggersInputTags = cms.VInputTag( )


### Get new RCT calibrations from CondDB until new GlobalTag is ready
### Goes with tauL1Calib_LUT.txt
### Need new GCT jet calibrations to go with it 
#from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
#process.rctSFDB = cms.ESSource("PoolDBESSource",
#    CondDBSetup,
#    connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
#    DumpStat=cms.untracked.bool(True),
#    toGet = cms.VPSet(
#        cms.PSet(
#            record = cms.string('L1RCTParametersRcd'),
#            tag = cms.string('L1RCTParametersRcd_L1TDevelCollisions_ExtendedScaleFactorsV4')
#        )
#    )
#)
## process.prefer("caloParmsDB")
#process.es_prefer_rctSFDB = cms.ESPrefer( "PoolDBESSource", "rctSFDB" )

## load the CaloStage1 params
## process.GlobalTag.toGet = cms.VPSet(
##   cms.PSet(record = cms.string("L1TCaloParamsRcd"),
##            tag = cms.string("L1TCaloParams_CRAFT09_hlt"),
##            connect = cms.string("sqlite:l1config.db")
##           )
## )

process.p1 = cms.Path(
    process.L1TCaloStage1_PPFromRaw
    +process.simGtDigis
    +process.l1ExtraLayer2
    )

process.output_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.p1, process.output_step
    )

# Spit out filter efficiency at the end.
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
