# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: l1 -s L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec --filein=/store/mc/Fall13dr/TT_Tune4C_13TeV-pythia8-tauola/GEN-SIM-RAW/tsg_PU40bx25_POSTLS162_V2-v1/00000/007939EF-8075-E311-B675-0025905938AA.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('L1')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_LS172_V15-v1/00000/08DC28C8-E059-E411-B9C6-0025905B85AA.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('l1 nevts:1'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring("keep *"),#process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('L1_stage1.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

# enable debug message logging for our modules
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('l1tCaloStage1Digis'),
    files = cms.untracked.PSet(
        critical = cms.untracked.PSet(

        ),
        detailedInfo = cms.untracked.PSet(
            threshold = cms.untracked.string('DEBUG')
        )
    )
)

# to get input digis
process.load('Configuration.StandardSequences.RawToDigi_cff')

# upgrade calo stage 1
process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_PPFromRaw_cff')
#process.load('L1Trigger.L1TCalorimeter.l1tStage1CaloAnalyzer_cfi')

# TTree output file
#process.load("CommonTools.UtilAlgos.TFileService_cfi")
#process.TFileService.fileName = cms.string('l1t.root')

# Path and EndPath definitions
process.L1simulation_step = cms.Path(
    process.ecalDigis
    +process.hcalDigis
    +process.L1TCaloStage1_PPFromRaw
)

process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.L1simulation_step,
                                process.RECOSIMoutput_step)

