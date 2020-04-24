import FWCore.ParameterSet.Config as cms

process = cms.Process('L1TEST')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('L1-O2O')
process.load('Configuration.EventContent.EventContent_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source('EmptySource',
    firstRun = cms.untracked.uint32(248518)
)

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'GR_P_V56')

process.GlobalTag.DumpStat = cms.untracked.bool(True)

process.l1tConfigDumper = cms.EDAnalyzer("L1TConfigDumper")

process.p = cms.Path(
    process.l1tConfigDumper
)
