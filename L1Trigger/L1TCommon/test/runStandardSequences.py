from __future__ import print_function
# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: debug --no_exec --conditions auto:run2_mc_25ns14e33_v4 -s DIGI:pdigi_valid,L1,DIGI2RAW,RAW2DIGI --datatier GEN-SIM-DIGI-RAW-HLTDEBUG -n 10 --era Run2_25ns --eventcontent FEVTDEBUGHLT --filein filelist:step1_dasquery.log --fileout file:step2.root
import FWCore.ParameterSet.Config as cms


#from Configuration.Eras.Era_Run2_25ns_cff import Run2_25ns
#process = cms.Process('L1SEQS',Run2_25ns)
from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
process = cms.Process('L1SEQS',Run2_2016)

process.MessageLogger = cms.Service(
    "MessageLogger",
    destinations = cms.untracked.vstring('l1tdebug','cerr'),
    l1tdebug = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),
    cerr = cms.untracked.PSet(threshold  = cms.untracked.string('WARNING')),
    debugModules = cms.untracked.vstring('*'))


# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
#     fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/g/gflouris/public/SingleMuPt6180_noanti_10k_eta1.root'),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/0A812333-427C-E511-A80A-0025905964A2.root', 
        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/1E9D9F9B-467C-E511-85B6-0025905A6090.root', 
        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/AA4FBC07-3E7C-E511-B9FC-00261894386C.root', 
        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/E2072991-3E7C-E511-803D-002618943947.root', 
        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/FAE20D9D-467C-E511-AF39-0025905B85D8.root'),
    inputCommands = cms.untracked.vstring('keep *', 
        'drop *_genParticles_*_*', 
        'drop *_genParticlesForJets_*_*', 
        'drop *_kt4GenJets_*_*', 
        'drop *_kt6GenJets_*_*', 
        'drop *_iterativeCone5GenJets_*_*', 
        'drop *_ak4GenJets_*_*', 
        'drop *_ak7GenJets_*_*', 
        'drop *_ak8GenJets_*_*', 
        'drop *_ak4GenJetsNoNu_*_*', 
        'drop *_ak8GenJetsNoNu_*_*', 
        'drop *_genCandidatesForMET_*_*', 
        'drop *_genParticlesForMETAllVisible_*_*', 
        'drop *_genMetCalo_*_*', 
        'drop *_genMetCaloAndNonPrompt_*_*', 
        'drop *_genMetTrue_*_*', 
        'drop *_genMetIC5GenJs_*_*'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('debug nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-HLTDEBUG'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(1048576),
    fileName = cms.untracked.string('file:step2.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.l1reco_step = cms.Path(process.L1Reco)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# additional tests:
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")
process.load('L1Trigger.L1TCommon.l1tSummaryStage2Digis_cfi')
process.load('L1Trigger.L1TCommon.l1tSummaryStage2SimDigis_cfi')

process.debug_step = cms.Path(
#    process.dumpES + 
#    process.dumpED +
    process.l1tSummaryStage2Digis +
    process.l1tSummaryStage2SimDigis
)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.raw2digi_step,process.l1reco_step,process.debug_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)

print("L1T Emulation Sequence is:  ")
print(process.SimL1Emulator)
#print "L1T DigiToRaw Sequence is:  "
#print process.L1TDigiToRaw
#print "L1T RawToDigi Sequence is:  "
#print process.L1TRawToDigi
#print "L1T Reco Sequence is:  "
#print process.L1Reco
print("DigiToRaw is:  ")
print(process.DigiToRaw)
