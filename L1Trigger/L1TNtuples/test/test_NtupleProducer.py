
import FWCore.ParameterSet.Config as cms

process = cms.Process("L1NTUPLE")

# conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond_condDBv2 import autoCond
process.GlobalTag.globaltag = cms.string( autoCond['run2_mc'] )

# input file
from L1Trigger.L1TNtuples.RelValInputFiles import *
process.source = cms.Source (
  "PoolSource",
  fileNames = cms.untracked.vstring( RelValInputFile_RAW() )
)

# N events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

# output file
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('L1Tree.root')
)

# producer under test
process.load("L1Trigger.L1TNtuples.l1NtupleProducer_cfi")
process.load("Configuration.StandardSequences.RawToDigi_cff")

process.p = cms.Path(
  process.RawToDigi
  +process.l1NtupleProducer
)

