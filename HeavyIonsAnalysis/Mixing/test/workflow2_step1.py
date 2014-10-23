# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --conditions auto:run1_mc -s DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@fake,RAW2DIGI,L1Reco --pileup_input file:ZmumuJets_Pt_20_300_GEN_8TeV_cfg_GEN_SIM.root --datatier GEN-SIM-DIGI-RAW-HLTDEBUG --pileup default -n 10 --eventcontent FEVTDEBUGHLT
import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_2012_Summer_50ns_PoissonOOTPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
#process.load('HLTrigger.Configuration.HLT_Fake_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('file:ZmumuJets_Pt_20_300_GEN_8TeV_cfg_GEN_SIM.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step2 nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(1048576),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    fileName = cms.untracked.string('step2_playbackonly.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-HLTDEBUG')
    )
)

# Additional output definition

# Other statements
#process.mix.input.fileNames = cms.untracked.vstring(['file:ZmumuJets_Pt_20_300_GEN_8TeV_cfg_GEN_SIM.root'])
process.mix.input.fileNames = cms.untracked.vstring(['/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/16547ECB-9C4B-E411-A815-0025905964BC.root', '/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/86C3C326-9F4B-E411-903D-0025905A48EC.root', '/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/C48D8223-9F4B-E411-BC37-0026189438DC.root', '/store/relval/CMSSW_7_2_0_pre7/RelValQCD_Pt_80_120_13/GEN-SIM/PRE_LS172_V11-v1/00000/D070AB62-9D4B-E411-9766-002618FDA207.root'])
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_mc', '')

process.p = cms.Path(process.mix)
process.outpath = cms.EndPath(process.FEVTDEBUGHLToutput)
