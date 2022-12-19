
import FWCore.ParameterSet.Config as cms

process = cms.Process("L1NTUPLE")

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


# conditions
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = cms.string( autoCond['run2_mc'] )

# input file
from L1Trigger.L1TNtuples.RelValInputFiles import *
process.source = cms.Source (
  "PoolSource",
  fileNames = cms.untracked.vstring( RelValInputFile_AOD() )
)

# N events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

# output file
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('L1Tree.root')
)

# producer under test
process.load("L1Trigger.L1TNtuples.l1JetRecoTreeProducer_cfi")

#process.l1RecoTreeProducer.jetTag = cms.untracked.InputTag("ak4PFCHSJetsL2")

process.p = cms.Path(
  process.l1JetRecoTreeProducer
)

