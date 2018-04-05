import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Services_cff")

## process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
## from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
## process.GlobalTag.globaltag = cms.string('GR_H_V44')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32(1) )

process.load('L1Trigger.L1TCalorimeter.caloStage1Params_cfi')
process.load("L1Trigger.L1TCalorimeter.caloStage1WriteLuts_cfi")


process.p = cms.Path(process.writeLuts)

